"""Best-effort progress telemetry, separate from mandatory evidence writes."""
import threading


class Telemetry:
    def __init__(self, publish, interval=30):
        self.publish, self.interval = publish, interval
        self.halt = threading.Event()
        self.errors = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.halt.wait(self.interval):
            try:
                self.publish()
            except Exception as error:
                # Rank/process/raw receipts are independently checked and required.
                self.errors[:] = [f"{type(error).__name__}: {error}"]

    def start(self):
        self.thread.start()
        return self

    def stop(self, timeout=10):
        self.halt.set()
        self.thread.join(timeout=timeout)
        return {"stopped": not self.thread.is_alive(), "last_errors": list(self.errors),
                "scope": "optional_progress_only"}
