# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Registry of audio analysis types for the `cxas trace audio analyze` command.

Each analysis declares (a) a stable name, (b) a Gemini prompt, and (c) a
file-filter that picks which audio files in a conversation it cares about
(e.g. `agent-turn-*.wav` vs `full-session.wav`). Conversations are stored in
GCS as a directory of files per conversation, so passing only the relevant
files keeps the Gemini call focused and cheap.

To add a new analysis: subclass `AudioAnalysis`, append an instance to
`_ALL_ANALYSES`, and the `ANALYSIS_REGISTRY` will pick it up automatically.

Prompt overrides can be supplied via `trace.yaml` under
`gemini.audio_metrics.<analysis_name>.prompt` — the runtime resolves overrides
in `Traces.analyze_audio` and falls back to the prompt defined here.
"""

import enum
from abc import ABC, abstractmethod


class AnalysisType(str, enum.Enum):
    """String-valued enum (compatible with Python 3.10+; `enum.StrEnum`
    is only available from 3.11)."""

    AGENT_VOICE_CONSISTENCY = "agent_voice_consistency"
    NO_LONG_PAUSES = "no_long_pauses"
    AGENT_HAVING_TROUBLE = "agent_having_trouble"
    AGENT_LOOPING = "agent_looping"
    AGENT_CUTOFF = "agent_cutoff"
    AGENT_TRANSCRIPT_MISMATCH = "agent_transcript_mismatch"

    def __str__(self) -> str:
        return self.value


class AudioAnalysis(ABC):
    """Base class for audio analyses."""

    @property
    @abstractmethod
    def name(self) -> AnalysisType:
        """Stable name; must match the AnalysisType enum value."""

    @property
    @abstractmethod
    def prompt(self) -> str:
        """The Gemini prompt to run on the filtered audio files."""

    @abstractmethod
    def filter_files(self, files_in_conversation: list[str]) -> list[str]:
        """Picks the subset of conversation audio files this analysis needs."""

    @property
    def check_instruction(self) -> str | None:
        """A brief 1-2 sentence instruction for this check to bundle into expectations evaluation."""
        return None


class VoiceConsistencyAnalysis(AudioAnalysis):
    """Analysis for voice consistency."""

    @property
    def name(self) -> AnalysisType:
        return AnalysisType.AGENT_VOICE_CONSISTENCY

    @property
    def prompt(self) -> str:
        return """
# Setup
- You are a voice analysis assistant that will analyze conversation audio/wav files.
- You will be given a list of audio clips / files. Each clip contains the audio of one person speaking.

# Task
- Your task is to determine if all the audio files are from the same speaker.
- You should analyze the voice profile (including pitch, timbre, and accent) to determine if the files are from the same speaker.
- For finding outlier audio profiles, look for distinct differences (e.g. male vs female voice / pitch, different timbres) rather than minor fluctuations.

# Output
- Report PASS if all audio files are from the same speaker and provide a justification.
- Report FAIL if any of the audio files are from a different speaker and provide a justification.
"""

    def filter_files(self, files_in_conversation: list[str]) -> list[str]:
        return [f for f in files_in_conversation if "agent-turn" in f]

    @property
    def check_instruction(self) -> str:
        return "Voice Consistency: Verify that the agent's voice profile (pitch, timbre, and accent) remains consistent across all turns, sounding like the same speaker."


class NoLongPausesAnalysis(AudioAnalysis):
    """Analysis for long pauses."""

    @property
    def name(self) -> AnalysisType:
        return AnalysisType.NO_LONG_PAUSES

    @property
    def prompt(self) -> str:
        return """
# Setup
- You are a voice analysis assistant that will analyze conversation audio/wav files.
- You will be given an audio clip of a conversation between two people.

# Task
- Your task is to determine if the conversation contains any pauses longer than 20 seconds.
- For finding pauses, look for distinct differences (e.g. silence or non-speech sounds like music or ambient noise) rather than minor fluctuations.
- If there is a pause between 0 and 19 seconds, that should not be considered a long pause.

# Output
- Report PASS if the audio clip does not contain long pauses and provide a justification.
- Report FAIL if the audio clip contains a long pause and provide a justification.
"""

    def filter_files(self, files_in_conversation: list[str]) -> list[str]:
        return [f for f in files_in_conversation if "full-session" in f]

    @property
    def check_instruction(self) -> str:
        return "No Long Pauses: Verify there are no silent intervals or pauses longer than 20 seconds in the conversation."


class AgentHavingTroubleAnalysis(AudioAnalysis):
    """Analysis for agent having trouble."""

    @property
    def name(self) -> AnalysisType:
        return AnalysisType.AGENT_HAVING_TROUBLE

    @property
    def prompt(self) -> str:
        return """
# Setup
- You are a voice analysis assistant that will analyze conversation audio/wav files.
- You will be given a list of audio clips / files. Each clip contains the audio of one person speaking.

# Task
- Your task is to determine if any of the audio clips contain a person saying they are having trouble.

# Output
- Report PASS if none of the audio files contain a person saying they are having trouble and provide a justification.
- Report FAIL if any of the audio files contain a person saying they are having trouble and provide a justification.
"""

    def filter_files(self, files_in_conversation: list[str]) -> list[str]:
        return [f for f in files_in_conversation if "agent-turn" in f]

    @property
    def check_instruction(self) -> str:
        return "Agent Success: Verify that the agent does not state it is having trouble or fail to understand basic inputs."


class AgentLoopingAnalysis(AudioAnalysis):
    """Analysis for agent looping."""

    @property
    def name(self) -> AnalysisType:
        return AnalysisType.AGENT_LOOPING

    @property
    def prompt(self) -> str:
        return """
# Setup
- You are a voice analysis assistant that will analyze conversation audio/wav files.
- You will be given a list of audio clips / files. All audio clips are of the virtual agent speaking in a conversation.
- A common issue with virtual agents is to get stuck in a loop, repeating the same sentence or phrase multiple times.

# Task
- Your task is to determine if the speaker repeats the same sentence or phrase multiple times in the conversation.
- The repetition could be exact or contain minor variations, but the sentiment should be the same. Questions and statements never have the same sentiment and should not be considered to repeat.
- The repetition could occur in the same file or across multiple files.

# Output
- Report PASS if none of the audio files contain a person repeating the same sentence or phrase multiple times and provide a justification.
- Report FAIL if any of the audio files contain a person repeating the same sentence or phrase multiple times and provide a justification.
"""

    def filter_files(self, files_in_conversation: list[str]) -> list[str]:
        return [f for f in files_in_conversation if "agent-turn" in f]

    @property
    def check_instruction(self) -> str:
        return "No Looping: Verify the agent does not repeat identical sentences or get stuck in a conversational loop."


class AgentCutoffAnalysis(AudioAnalysis):
    """Analysis for agent sentence cutoff."""

    @property
    def name(self) -> AnalysisType:
        return AnalysisType.AGENT_CUTOFF

    @property
    def prompt(self) -> str:
        return """
# Setup
- You are a voice analysis assistant that will analyze conversation audio/wav files.
- You will be given a list of audio clips / files. All audio clips are of the virtual agent speaking in a conversation.

# Task
- Your task is to determine if any of the agent's sentences are cutoff or abruptly ended before the sentence is completed.
- This could happen due to system issues or early termination of the turn.

# Output
- Report PASS if none of the audio files contain a sentence that is cutoff and provide a justification.
- Report FAIL if any of the audio files contain a sentence that is cutoff and provide a justification.
"""

    def filter_files(self, files_in_conversation: list[str]) -> list[str]:
        return [f for f in files_in_conversation if "agent-turn" in f]

    @property
    def check_instruction(self) -> str:
        return "No Cut-offs: Verify that no sentences are abruptly cut off or terminated before completion."


class TranscriptMismatchAnalysis(AudioAnalysis):
    """Analysis for checking if the agent's spoken audio matches the transcribed text."""

    @property
    def name(self) -> AnalysisType:
        return AnalysisType.AGENT_TRANSCRIPT_MISMATCH

    @property
    def prompt(self) -> str:
        return """
# Setup
- You are a voice analysis assistant that will analyze conversation audio/wav
  files and compare them against a transcription.
- You will be given a METADATA.json containing the transcribed conversation
  log, and a list of turn audio files (e.g. agent-turn-N.wav).

# Task
- Read the METADATA.json file to extract the expected transcribed text for
  each virtual agent turn.
- Listen carefully to each corresponding audio clip.
- Compare them using the following instruction:
  "The spoken audio in the turn must semantically match the text transcript.
  Ignore minor differences in wording, formatting (e.g., '1 8 0' vs 'one
  eight zero'), filler words, or contractions, as long as the core meaning,
  intent, and instructions are identical. Ignore the omission or addition
  of polite fillers, greetings, or sentences that explain why information is
  being requested (e.g., 'I need that so I can help you', 'to get started'),
  as long as the core request or instruction itself is present and correct.
  Flag as FAILED only if there is a semantic contradiction (e.g., 'required'
  vs 'not required'), a change in key information (like dates, identifiers,
  or service names), or if critical information is added or omitted in the
  audio that changes the meaning."

# Output Format
- If the comparison passes (all audio files match their corresponding
  transcribed text perfectly according to the rules), output:
  PASS
  <justification>
- If the comparison fails, output:
  FAIL
  If the audit FAILS, the justification MUST be formatted as a numbered list
  of issues. Each issue must reference the Turn number where it occurred and
  describe the mismatch clearly.
  Example:
  1. Turn 3: Audio omitted 'please retry'.
  2. Turn 5: Audio said '123' but text transcript shows '456'.
"""

    def filter_files(self, files_in_conversation: list[str]) -> list[str]:
        return [
            f
            for f in files_in_conversation
            if "agent-turn" in f or "METADATA.json" in f
        ]

    @property
    def check_instruction(self) -> str:
        return (
            "Transcript Mismatch: Verify that the agent's spoken audio "
            "accurately matches the transcribed text without cutoffs, "
            "truncated words, or semantic discrepancy."
        )


_ALL_ANALYSES: list[AudioAnalysis] = [
    VoiceConsistencyAnalysis(),
    NoLongPausesAnalysis(),
    AgentHavingTroubleAnalysis(),
    AgentLoopingAnalysis(),
    AgentCutoffAnalysis(),
    TranscriptMismatchAnalysis(),
]

ANALYSIS_REGISTRY: dict[str, AudioAnalysis] = {
    str(analysis.name): analysis for analysis in _ALL_ANALYSES
}
