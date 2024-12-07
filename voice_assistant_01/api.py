from typing import Annotated
import enum
from livekit.agents import llm
import logging

logger = logging.getLogger("temperature-control")
logger.setLevel(logging.INFO)


class Zone(enum.Enum):
    LIVING_ROOM = "living_room"
    BEDROOM = "bedroom"


class AssistantFnc(llm.FunctionContext):
    def __init__(self) -> None:
        super().__init__()

        self._temperature = {
            Zone.LIVING_ROOM: 25,
            Zone.BEDROOM: 28
        }

    @llm.ai_callable(description="This can be used to get temperatures in a specific room.")
    def get_temperature(self, zone: Annotated[Zone, llm.TypeInfo(description="The specific zone")]):
        logger.info(f"get temp - zone {zone}")
        temp = self._temperature[Zone(zone)]
        return f"The temperature at the {zone} is {temp}C"
