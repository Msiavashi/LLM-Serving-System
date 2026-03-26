import asyncio
import logging

from src.models.model_adapter import ModelAdapter
from src.engines.qllm_engine import QllmEngine
from src.cache.sequence_cache_manager import SequenceCacheManager
from src.samplers.sampling_params import SamplingParams
from src.schedulers.factory import SchedulerFactory
from src.config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class SchedulerService:
    def __init__(self):
        config = ConfigManager()
        logger.info(f"Initializing SchedulerService with {config.get('model.type')}...")

        # Load model via v2 ModelAdapter
        model_type = config.get('model.type', 'mixtral')
        rank = config.get('model.rank', 0)
        logger.info(f"Loading model: {model_type} on rank {rank}")

        self.adapter = ModelAdapter.from_name(model_type, rank=rank)

        # Inject queues if MoE
        if self.adapter.is_moe:
            self.adapter.inject_queues()

        # Create engine
        sampling_params = SamplingParams(
            eos_token_id=self.adapter.tokenizer.eos_token_id,
        )
        self.engine = QllmEngine(
            model_adapter=self.adapter,
            cache_manager=SequenceCacheManager(),
            sampling_params=sampling_params,
        )

        # Create scheduler
        scheduler_type = config.get('scheduler.type', 'fcfs')
        batch_size = config.get('scheduler.batch_size', 16)
        logger.info(f"Creating {scheduler_type} scheduler (batch_size={batch_size})")

        self.scheduler = SchedulerFactory.create_scheduler(
            name=scheduler_type,
            engine=self.engine,
            tokenizer=self.adapter.tokenizer,
            batch_size=batch_size,
        )

        if self.adapter.is_moe:
            self.adapter.set_scheduler(self.scheduler)

        logger.info("SchedulerService initialized.")

    async def run_scheduler(self):
        logger.info("Running scheduler...")
        try:
            await asyncio.gather(
                self.scheduler.run_scheduler()
            )
        except Exception as e:
            logger.error(f"Error in scheduler: {e}")
            raise


async def create_and_run_service():
    service = SchedulerService()
    await service.run_scheduler()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(create_and_run_service())
