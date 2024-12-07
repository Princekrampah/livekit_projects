import asyncio
from dotenv import load_dotenv


from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import anthropic, silero, openai


from api import AssistantFnc

# load env
load_dotenv()

# Code that triggers voice assistant


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a helpul voice AI assistant designed to help Prince.",
            "You should use short and concise responses, avoid usage of unpronoucable punctuations."
        )
    )

    # Specify that we want to connect only to audio tracks only at the moment
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    fnc_ctx = AssistantFnc()

    assistant = VoiceAssistant(
        # Voice Activity Detection(vad)
        # Specifies what model we are using to detect voice
        # To detect if the user is speaking
        vad=silero.VAD.load(),
        # Speech To Text (stt)
        stt=openai.STT(),
        llm=anthropic.LLM(),
        tts=openai.TTS(),
        chat_ctx=initial_ctx,
        fnc_ctx=fnc_ctx
    )

    # We can have multiple rooms and these voice assistants can connect
    # to one or multiple rooms

    # 1. Agent connects to a livekit server
    # 2. Livekit server send a job to the agent
    # 3. The job sent has a room associated with it
    # So below we are just subscribing to the audio inside that room the job came with
    assistant.start(ctx.room)

    await asyncio.sleep(1)
    await assistant.say("Hello, how can I help you today!", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
