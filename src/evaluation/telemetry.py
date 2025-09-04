"""
Telemetry logging system for user interactions and implicit feedback.
"""
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import threading
from queue import Queue
import logging


class InteractionType(Enum):
    """Types of user interactions."""
    QUERY = "query"
    RESPONSE = "response"
    CLICK = "click"
    HOVER = "hover"
    SCROLL = "scroll"
    COPY = "copy"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    CONVERSATION_START = "conversation_start"
    CONVERSATION_END = "conversation_end"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    ABANDONMENT = "abandonment"
    FOLLOW_UP = "follow_up"


class EventType(Enum):
    """Types of telemetry events."""
    USER_INTERACTION = "user_interaction"
    SYSTEM_EVENT = "system_event"
    PERFORMANCE_METRIC = "performance_metric"
    ERROR_EVENT = "error_event"


@dataclass
class UserInteraction:
    """Represents a user interaction event."""
    event_id: str
    session_id: str
    conversation_id: str
    turn_id: str
    user_id: Optional[str]
    interaction_type: InteractionType
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "user_id": self.user_id,
            "interaction_type": self.interaction_type.value,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "metadata": self.metadata
        }


class SessionTracker:
    """Tracks user sessions and conversations."""
    
    def __init__(self):
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        self.active_turns: Dict[str, Dict[str, Any]] = {}
    
    def start_session(self, user_id: Optional[str] = None) -> str:
        """Start a new session."""
        session_id = str(uuid.uuid4())
        self.active_sessions[session_id] = {
            "user_id": user_id,
            "start_time": datetime.now(timezone.utc),
            "last_activity": datetime.now(timezone.utc),
            "interaction_count": 0
        }
        return session_id
    
    def end_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """End a session and return session data."""
        if session_id in self.active_sessions:
            session_data = self.active_sessions[session_id].copy()
            session_data["end_time"] = datetime.now(timezone.utc)
            session_data["duration"] = (session_data["end_time"] - session_data["start_time"]).total_seconds()
            del self.active_sessions[session_id]
            return session_data
        return None
    
    def start_conversation(self, session_id: str) -> str:
        """Start a new conversation within a session."""
        conversation_id = str(uuid.uuid4())
        self.active_conversations[conversation_id] = {
            "session_id": session_id,
            "start_time": datetime.now(timezone.utc),
            "last_activity": datetime.now(timezone.utc),
            "turn_count": 0
        }
        return conversation_id
    
    def end_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """End a conversation and return conversation data."""
        if conversation_id in self.active_conversations:
            conversation_data = self.active_conversations[conversation_id].copy()
            conversation_data["end_time"] = datetime.now(timezone.utc)
            conversation_data["duration"] = (conversation_data["end_time"] - conversation_data["start_time"]).total_seconds()
            del self.active_conversations[conversation_id]
            return conversation_data
        return None
    
    def start_turn(self, conversation_id: str) -> str:
        """Start a new turn within a conversation."""
        turn_id = str(uuid.uuid4())
        self.active_turns[turn_id] = {
            "conversation_id": conversation_id,
            "start_time": datetime.now(timezone.utc),
            "last_activity": datetime.now(timezone.utc)
        }
        
        # Update conversation turn count
        if conversation_id in self.active_conversations:
            self.active_conversations[conversation_id]["turn_count"] += 1
            self.active_conversations[conversation_id]["last_activity"] = datetime.now(timezone.utc)
        
        return turn_id
    
    def end_turn(self, turn_id: str) -> Optional[Dict[str, Any]]:
        """End a turn and return turn data."""
        if turn_id in self.active_turns:
            turn_data = self.active_turns[turn_id].copy()
            turn_data["end_time"] = datetime.now(timezone.utc)
            turn_data["duration"] = (turn_data["end_time"] - turn_data["start_time"]).total_seconds()
            del self.active_turns[turn_id]
            return turn_data
        return None
    
    def update_activity(self, session_id: str, conversation_id: Optional[str] = None, turn_id: Optional[str] = None):
        """Update last activity timestamp."""
        current_time = datetime.now(timezone.utc)
        
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["last_activity"] = current_time
            self.active_sessions[session_id]["interaction_count"] += 1
        
        if conversation_id and conversation_id in self.active_conversations:
            self.active_conversations[conversation_id]["last_activity"] = current_time
        
        if turn_id and turn_id in self.active_turns:
            self.active_turns[turn_id]["last_activity"] = current_time


class TelemetryLogger:
    """Main telemetry logging system."""
    
    def __init__(self, batch_size: int = 100, flush_interval: float = 30.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.event_queue: Queue = Queue()
        self.session_tracker = SessionTracker()
        self.logger = logging.getLogger(__name__)
        
        # Start background thread for processing events
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._process_events, daemon=True)
        self._worker_thread.start()
    
    def log_query(
        self,
        session_id: str,
        conversation_id: str,
        turn_id: str,
        query: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log a user query."""
        event_id = str(uuid.uuid4())
        
        interaction = UserInteraction(
            event_id=event_id,
            session_id=session_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            user_id=user_id,
            interaction_type=InteractionType.QUERY,
            event_type=EventType.USER_INTERACTION,
            timestamp=datetime.now(timezone.utc),
            data={
                "query": query,
                "query_length": len(query),
                "word_count": len(query.split())
            },
            metadata=metadata or {}
        )
        
        self._queue_event(interaction)
        self.session_tracker.update_activity(session_id, conversation_id, turn_id)
        return event_id
    
    def log_response(
        self,
        session_id: str,
        conversation_id: str,
        turn_id: str,
        response: str,
        response_time: float,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log a system response."""
        event_id = str(uuid.uuid4())
        
        interaction = UserInteraction(
            event_id=event_id,
            session_id=session_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            user_id=user_id,
            interaction_type=InteractionType.RESPONSE,
            event_type=EventType.SYSTEM_EVENT,
            timestamp=datetime.now(timezone.utc),
            data={
                "response": response,
                "response_length": len(response),
                "response_time": response_time,
                "word_count": len(response.split())
            },
            metadata=metadata or {}
        )
        
        self._queue_event(interaction)
        return event_id
    
    def log_click(
        self,
        session_id: str,
        conversation_id: str,
        turn_id: str,
        element: str,
        position: Dict[str, int],
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log a click interaction."""
        event_id = str(uuid.uuid4())
        
        interaction = UserInteraction(
            event_id=event_id,
            session_id=session_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            user_id=user_id,
            interaction_type=InteractionType.CLICK,
            event_type=EventType.USER_INTERACTION,
            timestamp=datetime.now(timezone.utc),
            data={
                "element": element,
                "position": position,
                "click_type": "left_click"  # Could be extended
            },
            metadata=metadata or {}
        )
        
        self._queue_event(interaction)
        self.session_tracker.update_activity(session_id, conversation_id, turn_id)
        return event_id
    
    def log_hover(
        self,
        session_id: str,
        conversation_id: str,
        turn_id: str,
        element: str,
        duration: float,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log a hover interaction."""
        event_id = str(uuid.uuid4())
        
        interaction = UserInteraction(
            event_id=event_id,
            session_id=session_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            user_id=user_id,
            interaction_type=InteractionType.HOVER,
            event_type=EventType.USER_INTERACTION,
            timestamp=datetime.now(timezone.utc),
            data={
                "element": element,
                "duration": duration
            },
            metadata=metadata or {}
        )
        
        self._queue_event(interaction)
        return event_id
    
    def log_scroll(
        self,
        session_id: str,
        conversation_id: str,
        turn_id: str,
        scroll_depth: float,
        scroll_direction: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log a scroll interaction."""
        event_id = str(uuid.uuid4())
        
        interaction = UserInteraction(
            event_id=event_id,
            session_id=session_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            user_id=user_id,
            interaction_type=InteractionType.SCROLL,
            event_type=EventType.USER_INTERACTION,
            timestamp=datetime.now(timezone.utc),
            data={
                "scroll_depth": scroll_depth,
                "scroll_direction": scroll_direction
            },
            metadata=metadata or {}
        )
        
        self._queue_event(interaction)
        self.session_tracker.update_activity(session_id, conversation_id, turn_id)
        return event_id
    
    def log_copy(
        self,
        session_id: str,
        conversation_id: str,
        turn_id: str,
        copied_text: str,
        source_element: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log a copy interaction."""
        event_id = str(uuid.uuid4())
        
        interaction = UserInteraction(
            event_id=event_id,
            session_id=session_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            user_id=user_id,
            interaction_type=InteractionType.COPY,
            event_type=EventType.USER_INTERACTION,
            timestamp=datetime.now(timezone.utc),
            data={
                "copied_text": copied_text,
                "copied_length": len(copied_text),
                "source_element": source_element
            },
            metadata=metadata or {}
        )
        
        self._queue_event(interaction)
        self.session_tracker.update_activity(session_id, conversation_id, turn_id)
        return event_id
    
    def log_dwell_time(
        self,
        session_id: str,
        conversation_id: str,
        turn_id: str,
        element: str,
        dwell_time: float,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log dwell time on an element."""
        event_id = str(uuid.uuid4())
        
        interaction = UserInteraction(
            event_id=event_id,
            session_id=session_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            user_id=user_id,
            interaction_type=InteractionType.HOVER,  # Using hover for dwell time
            event_type=EventType.USER_INTERACTION,
            timestamp=datetime.now(timezone.utc),
            data={
                "element": element,
                "dwell_time": dwell_time
            },
            metadata=metadata or {}
        )
        
        self._queue_event(interaction)
        return event_id
    
    def log_abandonment(
        self,
        session_id: str,
        conversation_id: str,
        turn_id: str,
        abandonment_type: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log session or conversation abandonment."""
        event_id = str(uuid.uuid4())
        
        interaction = UserInteraction(
            event_id=event_id,
            session_id=session_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            user_id=user_id,
            interaction_type=InteractionType.ABANDONMENT,
            event_type=EventType.USER_INTERACTION,
            timestamp=datetime.now(timezone.utc),
            data={
                "abandonment_type": abandonment_type
            },
            metadata=metadata or {}
        )
        
        self._queue_event(interaction)
        return event_id
    
    def log_follow_up(
        self,
        session_id: str,
        conversation_id: str,
        turn_id: str,
        follow_up_type: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log a follow-up interaction."""
        event_id = str(uuid.uuid4())
        
        interaction = UserInteraction(
            event_id=event_id,
            session_id=session_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            user_id=user_id,
            interaction_type=InteractionType.FOLLOW_UP,
            event_type=EventType.USER_INTERACTION,
            timestamp=datetime.now(timezone.utc),
            data={
                "follow_up_type": follow_up_type
            },
            metadata=metadata or {}
        )
        
        self._queue_event(interaction)
        self.session_tracker.update_activity(session_id, conversation_id, turn_id)
        return event_id
    
    def log_performance_metric(
        self,
        session_id: str,
        conversation_id: str,
        turn_id: str,
        metric_name: str,
        metric_value: float,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log a performance metric."""
        event_id = str(uuid.uuid4())
        
        interaction = UserInteraction(
            event_id=event_id,
            session_id=session_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            user_id=user_id,
            interaction_type=InteractionType.RESPONSE,  # Using response for performance
            event_type=EventType.PERFORMANCE_METRIC,
            timestamp=datetime.now(timezone.utc),
            data={
                "metric_name": metric_name,
                "metric_value": metric_value
            },
            metadata=metadata or {}
        )
        
        self._queue_event(interaction)
        return event_id
    
    def _queue_event(self, interaction: UserInteraction):
        """Queue an event for processing."""
        self.event_queue.put(interaction)
    
    def _process_events(self):
        """Background thread to process events."""
        batch = []
        last_flush = time.time()
        
        while not self._stop_event.is_set():
            try:
                # Try to get an event with timeout
                try:
                    interaction = self.event_queue.get(timeout=1.0)
                    batch.append(interaction)
                except:
                    # Timeout, check if we need to flush
                    if batch and (time.time() - last_flush) > self.flush_interval:
                        self._flush_batch(batch)
                        batch = []
                        last_flush = time.time()
                    continue
                
                # Check if batch is full
                if len(batch) >= self.batch_size:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()
                
            except Exception as e:
                self.logger.error(f"Error processing telemetry events: {e}")
    
    def _flush_batch(self, batch: List[UserInteraction]):
        """Flush a batch of events (to be implemented with actual storage)."""
        # This would typically write to Delta tables or other storage
        for interaction in batch:
            self.logger.info(f"Telemetry event: {interaction.to_dict()}")
    
    def flush(self):
        """Manually flush all pending events."""
        batch = []
        while not self.event_queue.empty():
            try:
                interaction = self.event_queue.get_nowait()
                batch.append(interaction)
            except:
                break
        
        if batch:
            self._flush_batch(batch)
    
    def stop(self):
        """Stop the telemetry logger."""
        self._stop_event.set()
        self.flush()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)


# Global telemetry logger instance
telemetry_logger = TelemetryLogger()
