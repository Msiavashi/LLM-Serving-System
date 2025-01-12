import asyncio
from .factory import ServiceFactory

async def create_and_run_service():
    service = ServiceFactory.create_service()
    await service.run_scheduler()

def run_service():
    asyncio.run(create_and_run_service())

if __name__ == "__main__":
    print("Starting SchedulerService...")
    run_service()
