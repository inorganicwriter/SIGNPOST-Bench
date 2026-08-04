import math


class MetricCalculator:
    # WLA exponential decay parameter
    # WLA = exp(-alpha * d), where d is error in km
    # alpha=0.005 gives: 1km→0.995, 25km→0.882, 200km→0.368, 750km→0.024, 2500km→0.000
    WLA_ALPHA = 0.005

    # Legacy thresholds (kept for reference / threshold-based analysis)
    WLA_THRESHOLDS = [1, 25, 200, 750, 2500]

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance (km) between two points.
        """
        try:
            # Convert decimal degrees to radians
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

            # Haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            c = 2 * math.asin(math.sqrt(a))
            r = 6371  # Radius of earth in kilometers
            return c * r
        except Exception:
            return None

    @staticmethod
    def calculate_wla(error_km, alpha=None):
        """
        Weighted Localization Accuracy (WLA) — Exponential decay.

        WLA = exp(-α · d)

        where d = error in km, α controls decay speed.
        Higher precision (smaller d) → score close to 1.0
        Lower precision (larger d)  → score decays exponentially toward 0.0

        Reference points (α=0.005):
            1 km   → 0.995
            25 km  → 0.882
            100 km → 0.607
            200 km → 0.368
            750 km → 0.024
            2500 km → 0.000

        Args:
            error_km: Prediction error in kilometers (None → 0.0)
            alpha: Decay rate (default: MetricCalculator.WLA_ALPHA = 0.005)

        Returns:
            WLA score in [0, 1]
        """
        if error_km is None:
            return 0.0
        if alpha is None:
            alpha = MetricCalculator.WLA_ALPHA
        return math.exp(-alpha * error_km)

    @staticmethod
    def calculate_tbs(clean_error, adversarial_error):
        """
        Text Bias Score (TBS)
        TBS = Error_Adv - Error_Clean
        """
        if clean_error is None or adversarial_error is None:
            return None
        return adversarial_error - clean_error

    @staticmethod
    def calculate_tfr(pred_lat, pred_lon, trap_lat, trap_lon, threshold_km=50):
        """
        Trap-Fit Rate (TFR) hit check
        Checks if prediction is within threshold_km of the trap location.
        """
        if pred_lat is None or trap_lat is None:
            return False

        dist = MetricCalculator.haversine_distance(pred_lat, pred_lon, trap_lat, trap_lon)
        if dist is None:
            return False

        return dist < threshold_km

    # ===================================================================
    #  Conflict Probing Metrics (for mechanism analysis)
    # ===================================================================

    @staticmethod
    def calculate_cda(probing_results):
        """
        Conflict Detection Accuracy (CDA).
        Proportion of conflict samples where model correctly identifies
        inconsistency (consistent=false).

        Args:
            probing_results: list of dicts with 'consistent' field (bool or None)

        Returns:
            (cda_rate, n_valid) tuple
        """
        valid = [r for r in probing_results if r.get("consistent") is not None]
        if not valid:
            return 0.0, 0
        detected = sum(1 for r in valid if r["consistent"] is False)
        return detected / len(valid), len(valid)

    @staticmethod
    def calculate_mpr(probing_results):
        """
        Modality Preference Rate (MPR).
        Distribution of trusted_source choices across samples.

        Returns:
            dict with keys "Visual", "Textual", "Both", "Unknown" -> count and rate
        """
        counts = {"Visual": 0, "Textual": 0, "Both": 0, "Unknown": 0}
        for r in probing_results:
            src = r.get("trusted_source", "Unknown") or "Unknown"
            if src not in counts:
                src = "Unknown"
            counts[src] += 1
        total = sum(counts.values())
        rates = {k: v / total if total > 0 else 0.0 for k, v in counts.items()}
        return {"counts": counts, "rates": rates, "total": total}

    @staticmethod
    def calculate_rcs(probing_results):
        """
        Reasoning Consistency Score (RCS).
        Proportion of samples where the model's stated trusted_source
        is consistent with its actual prediction behavior.

        Returns:
            (rcs_rate, n_valid) tuple
        """
        valid = [r for r in probing_results if r.get("rcs_consistent") is not None]
        if not valid:
            return 0.0, 0
        consistent = sum(1 for r in valid if r["rcs_consistent"])
        return consistent / len(valid), len(valid)

    @staticmethod
    def calculate_caa(probing_results):
        """
        Conflict-Aware Accuracy (CAA).
        Mean WLA on the subset where model detected conflict (cda_hit=True).

        Returns:
            (caa_wla, n_detected) tuple
        """
        detected = [r for r in probing_results if r.get("cda_hit") is True and r.get("error_km") is not None]
        if not detected:
            return 0.0, 0
        mean_wla = sum(r.get("wla_score", 0) for r in detected) / len(detected)
        return mean_wla, len(detected)

    @staticmethod
    def calculate_tpg(probing_results, trap_threshold_km=50):
        """
        Trap Preference Gap (TPG).
        Difference in trap-fall rate between conflict samples and blank samples.
        TPG = TFR_conflict - TFR_blank

        Args:
            probing_results: list of dicts, must include 'attack_type' field

        Returns:
            (tpg, tfr_conflict, tfr_blank) tuple
        """
        conflict = [
            r
            for r in probing_results
            if r.get("attack_type") not in ("blank", "original", "clean", None) and r.get("pred_lat") is not None
        ]
        blank = [r for r in probing_results if r.get("attack_type") == "blank" and r.get("pred_lat") is not None]

        def tfr(samples):
            if not samples:
                return 0.0
            hits = 0
            for r in samples:
                trap_lat = r.get("trap_lat")
                trap_lon = r.get("trap_lon")
                if trap_lat is not None and trap_lon is not None:
                    dist = MetricCalculator.haversine_distance(r["pred_lat"], r["pred_lon"], trap_lat, trap_lon)
                    if dist is not None and dist < trap_threshold_km:
                        hits += 1
            return hits / len(samples)

        tfr_c = tfr(conflict)
        tfr_b = tfr(blank)
        return tfr_c - tfr_b, tfr_c, tfr_b

    @staticmethod
    def calculate_csg(probing_results):
        """
        Conflict Sensitivity Gap (CSG).
        Accuracy gap between detected-conflict and undetected-conflict subsets.
        CSG = WLA_detected - WLA_undetected

        Positive CSG means detecting conflict actually helps accuracy.
        """
        detected = [r for r in probing_results if r.get("cda_hit") is True and r.get("error_km") is not None]
        undetected = [r for r in probing_results if r.get("consistent") is True and r.get("error_km") is not None]

        wla_det = sum(r.get("wla_score", 0) for r in detected) / len(detected) if detected else 0.0
        wla_und = sum(r.get("wla_score", 0) for r in undetected) / len(undetected) if undetected else 0.0

        return wla_det - wla_und, wla_det, wla_und, len(detected), len(undetected)
