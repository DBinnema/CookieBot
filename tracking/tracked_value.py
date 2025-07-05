class TrackedValue:
    def __init__(self, number_region, suffix_region, label="Value", debug=False,
                 filter_config=None):
        """
        Tracks and filters an OCR-read numeric value with a suffix.

        Args:
            number_region: Screen region to read number from.
            suffix_region: Screen region to read suffix from.
            label: Debug label for logging.
            debug: Whether to print debug info.
            filter_config: Optional dict to customize filter (maxlen, deviation, etc).
        """
        self.region_num = number_region
        self.region_suffix = suffix_region
        self.label = label
        self.debug = debug

        # Allow override of filter config if needed
        self.filter = AdaptiveMeanFilter(**(filter_config or {}))

    def update(self):
        """
        Performs an OCR read + filtering. Returns the filtered result.
        """
        try:
            raw = interpret_number_with_suffix(
                self.region_num, self.region_suffix, debug=self.debug
            )

            if raw is None:
                print(f"[OCR] {self.label} fallback to last known value")
                return self.filter.last()

            print(f"[OCR] {self.label} raw before filter: {raw:.2f}")
            return self.filter.filter(raw)

        except Exception as e:
            print(f"[OCR ERROR] Exception in {self.label}: {e}")
            return self.filter.last()

    def last(self):
        return self.filter.last()
