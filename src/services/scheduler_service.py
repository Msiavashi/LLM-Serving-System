import asyncio
from src.engines.factory import EngineFactory
from src.schedulers.factory import SchedulerFactory
from src.models.model_factory import ModelFactory
from src.config.config_manager import ConfigManager

class SchedulerService:
    def __init__(self):
        config = ConfigManager()
        print(f"Initializing SchedulerService with {config.get('model.type')}...")
        
        # Initialize model, engine and scheduler
        print("Creating model instance and tokenizer...")
        model_instances, tokenizer = ModelFactory.create_model(
            model_type=config.get('model.type'),
            rank=config.get('model.rank', 0),
            **config.get('model.params', {})
        )
        model = model_instances[0].model

        print(f"Creating {config.get('engine.type')} engine...")
        self.engine = EngineFactory.create_engine(
            config.get('engine.type'),
            model=model
        )

        print(f"Creating {config.get('scheduler.type')} scheduler...")
        self.scheduler = SchedulerFactory.create_scheduler(
            name=config.get('scheduler.type'),
            engine=self.engine,
            tokenizer=tokenizer,
            batch_size=config.get('scheduler.batch_size')
        )
        print("SchedulerService initialized.")
        
        # Set the scheduler into the model through the new setter method.
        if hasattr(model, 'set_scheduler'):
            model.set_scheduler(self.scheduler)



    async def run_scheduler(self):
        print("Running scheduler...")
        try:
            await asyncio.gather(
                self.scheduler.run_scheduler()
            )
        except Exception as e:
            print(f"Error in scheduler: {e}")
            raise

async def create_and_run_service():
    service = SchedulerService()
    await service.run_scheduler()

if __name__ == "__main__":
    print("Starting SchedulerService...")
    asyncio.run(create_and_run_service())
