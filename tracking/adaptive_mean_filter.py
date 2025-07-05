from collections import deque

class AdaptiveMeanFilter:
    def __init__(self, maxlen=5, max_deviation_ratio=0.5, confirm_threshold=2):

        self.history = deque(maxlen=maxlen)
        self.last_value = None
        self.last_suspect = None
        self.suspect_count = 0
        self.max_deviation_ratio = max_deviation_ratio
        self.confirm_threshold = confirm_threshold


    def mean(self):
        return sum(self.history) / len(self.history) if self.history else 0.0

    def is_acceptable(self, new_val):
        if not self.history:
            return True  # Accept first reading

        avg = self.mean()
        deviation = abs(new_val - avg)
        return deviation <= (avg * self.max_deviation_ratio)

    def filter(self, new_val):
        """Returns the filtered value — either the new_val if accepted, or the last good one."""
        if self.is_acceptable(new_val):
            self.history.append(new_val)
            self.last_suspect = None
            self.suspect_count = 0
            self.last_value = new_val
            return new_val

        # Value is a suspect (not within expected range)
        if self.last_suspect is not None and abs(new_val - self.last_suspect) < self.mean() * self.max_deviation_ratio:
            self.suspect_count += 1
        else:
            self.last_suspect = new_val
            self.suspect_count = 1

        if self.suspect_count >= self.confirm_threshold:
            print(f"[FILTER] Accepted sustained new value after {self.confirm_threshold} attempts: {new_val}")
            self.history.append(new_val)
            self.last_suspect = None
            self.suspect_count = 0
            self.last_value = new_val
            return new_val
        else:
            print(f"[FILTER] Rejected transient spike/drop: {new_val}")
            return self.history[-1] if self.history else new_val

    def last(self):
        return self.last_value
