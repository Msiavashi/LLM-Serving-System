from .scheduler_service import SchedulerService

class ServiceFactory:
    @staticmethod
    def create_service() -> SchedulerService:
        return SchedulerService()
