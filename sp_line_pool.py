"""Subprocess Work pool for maskverif backend."""

import abc
import asyncio
from asyncio.subprocess import PIPE
import json
import logging
from typing import Iterator, Tuple, AsyncIterator, Awaitable
import threading
from queue import Queue


class LineAsyncWorker:
    def __init__(self, idx: int, proc: asyncio.subprocess.Process):
        self._idx = idx
        self._proc = proc

    @classmethod
    async def create(cls, idx: int, path: str, *args, **kwargs):
        proc = await asyncio.create_subprocess_exec(path, stdin=PIPE, stdout=PIPE)
        worker = cls(idx, proc)
        await worker.post_init(*args, **kwargs)
        return worker

    @abc.abstractmethod
    async def post_init(self, *args, **kwargs):
        pass

    async def write_line(self, line: str):
        self._proc.stdin.write((line + "\n").encode())
        await self._proc.stdin.drain()

    async def read_line(self):
        return (await self._proc.stdout.readline()).decode().rstrip("\n")

    async def execute_job(self, job):
        await self.write_line(job)
        return await self.read_line()

    async def terminate(self):
        self._proc.terminate()
        await self._proc.wait()


class WorkPool:
    def __init__(self, workers):
        self._workers = workers
        self._available_workers = asyncio.Queue()
        for worker in workers:
            self._available_workers.put_nowait(worker)

    @classmethod
    async def create(cls, num_workers: int, worker_factory) -> "WorkPool":
        workers = await asyncio.gather(*(worker_factory(i) for i in range(num_workers)))
        return cls(workers)

    async def submit(self, job) -> Awaitable:
        worker = await self._available_workers.get()
        future = asyncio.get_event_loop().create_future()

        async def _run_job():
            try:
                result = await worker.execute_job(job)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            finally:
                await self._available_workers.put(worker)

        asyncio.create_task(_run_job())
        return future

    async def exec(self, job):
        future = await self.submit(job)
        return await future

    async def map(self, jobs: AsyncIterator) -> AsyncIterator:
        queue: asyncio.PriorityQueue[Tuple[int, Awaitable]] = asyncio.PriorityQueue()

        async def producer():
            idx = 0
            async for job in jobs:
                future = await self.submit(job)
                await queue.put((idx, future))
                idx += 1
            await queue.put((float("inf"), None))  # Sentinel to signal end

        asyncio.create_task(producer())

        while True:
            _, future = await queue.get()
            if future is None:
                break  # Sentinel received — we're done
            result = await future
            yield result

    async def shutdown(self):
        await asyncio.gather(*(worker.terminate() for worker in self._workers))


class SyncWorkPool:
    def __init__(self, num_workers: int, worker_factory):
        # Could be 1, take a bit larger to minimize synchronization overheads.
        self._max_pending = num_workers
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._pool: WorkPool = self._submit_to_loop(
            WorkPool.create(num_workers, worker_factory)
        ).result()

    def _submit_to_loop(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def exec(self, job):
        return self._submit_to_loop(self._pool.exec(job)).result()

    def map(self, jobs: Iterator) -> Iterator:
        output_queue = Queue(self._max_pending)

        async def async_job_gen():
            for job in jobs:
                yield job

        async def run_map():
            async for result in self._pool.map(async_job_gen()):
                output_queue.put(result)
            output_queue.put(None)

        self._submit_to_loop(run_map())

        while True:
            result = output_queue.get()
            if result is None:
                break
            yield result

    def shutdown(self):
        self._submit_to_loop(self._pool.shutdown()).result()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
