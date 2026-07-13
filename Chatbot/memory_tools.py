"""
Chatbot/memory_tools.py

Defines the two tools the model can call to manage long-term memory:
add_memory (write a freeform note) and forget_memory (delete one).

Design note: these tools need to know WHICH profile to write to, but the
model itself should never have to supply a profile_id argument - that's
app context, not something the LLM should be guessing at. So instead of
plain @tool-decorated module-level functions, this file exposes a
FACTORY function, build_memory_tools(profile_id, thread_id), that closes
over those two values and returns tool objects bound to them. Whoever
builds the LLM call for a given turn (Chatbot/graph.py) calls this once
per turn with the current config's profile_id/thread_id, so the model
only ever sees the args it should actually decide: `note` and `note_id`.
"""
from langchain_core.tools import tool
from DB.long_term_memory import add_note, list_notes, delete_note
from DB.profiles import touch_profile

def build_memory_tools(profile_id: str, thread_id: str | None = None):
    """
    Returns a list of tool objects bound to the given profile/thread.
    Call this fresh each turn (profile can change between turns if the
    user switches profiles mid-session) and pass the result to
    llm.bind_tools(...).
    """
    @tool
    def add_memory(note: str) -> str:
        """
        Save a piece of information worth remembering about the user for
        future conversations, even in a different chat than this one.

        Use this when the user shares a durable fact, preference, or
        instruction - for example their name, job, likes/dislikes, goals,
        or an explicit request like "remember that..." or "note that...".
        Always call this when the user explicitly asks you to remember
        something. Otherwise, use your judgment: save things that would
        genuinely help you in a future conversation, not routine chat.

        Do not save sensitive information (passwords, financial details,
        health details) unless the user is clearly and explicitly asking
        you to store exactly that.

        Args:
            note: A short, self-contained statement of the fact to
                  remember, written so it makes sense read on its own
                  later, out of context (e.g. "User's name is Sadman",
                  not "their name is that").
        """
        try:
            note_id = add_note(profile_id, note, source_thread_id=thread_id)
            touch_profile(profile_id)
            return f"Saved to long-term memory (id: {note_id})."
        except Exception as e:
            return f"Failed to save memory: {e}"
        
    @tool
    def forget_memory(note_id: str) -> str:
        """
        Delete a previously saved memory note, when the user says
        something is no longer true, asks you to forget it, or corrects
        information you saved earlier.

        You must use the exact note id shown in your long-term memory
        context (each note is listed with its id). If you're not sure
        which note the user means, ask them to clarify rather than
        guessing - deleting the wrong note is worse than asking.

        Args:
            note_id: The exact id of the note to delete.
        """
        try:
            delete_note(profile_id, note_id)
            touch_profile(profile_id)
            return f"Deleted memory {note_id}."
        except Exception as e:
            return f"Failed to delete memory: {e}"

    return [add_memory, forget_memory]

def notes_as_context_with_ids(profile_id: str) -> str:
    """
    Like DB.long_term_memory.notes_as_context, but includes each note's id
    inline - needed so the model can actually call forget_memory(note_id)
    on something specific, since add_memory doesn't let it choose the id
    itself (ids are generated server-side as uuid4s).

    This is intentionally a separate function from the plain-text version
    in DB/long_term_memory.py: the UI's "What I remember" panel wants
    clean text without ids cluttering the display, but the model's system
    message needs ids to make forget_memory usable at all.
    """
    notes = list_notes(profile_id)
    if not notes:
        return ""

    lines = [f"- [id: {n['key']}] {n['text']}" for n in notes]
    return (
        "Long-term memory - things you know about this user from past "
        "conversations. Use forget_memory with the exact id shown if the "
        "user tells you something here is outdated or wrong:\n"
        + "\n".join(lines)
    )