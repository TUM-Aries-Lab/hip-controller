"""Trigger-driven baseline removal: subtracts the hip angle's resting offset.

The IMU angle carries a DC offset that depends on the subject and on how the
sensor is strapped. This module removes it the way the deployed Simulink
implementation does (``getHipKinematics_IMU.m``): average the angle over a short
window and subtract that mean from every later sample.

What matters is *when* the window is taken. The offset is only meaningful if it
is captured while the limb is in the reference posture, so collection is started
by an explicit trigger -- on the exosuit, the main switch (motor enable), whose
rising edge both integrations already detect. It is re-triggerable at any time,
without disturbing the filter state of the surrounding pipeline, which is what
makes it usable after a strap slips or between trials.

Two safeguards sit on top of the Simulink behaviour:

* **Stand-still gating.** A window is only accepted while the raw velocity stays
  below :attr:`BaselineRemovalConfig.max_velocity_rad_per_sec`. Triggering
  mid-stride restarts the window instead of averaging a moving leg.
* **No fabricated output.** The offset defaults to zero and the angle is passed
  through untouched until a window completes. Nothing downstream ever sees a
  forced value, so the stateful filters are not fed a step edge.

Nothing runs on its own. The main switch is recorded alongside the sensor data,
so CSV playback replays the operator's real trigger and reproduces the offset the
session actually ran with -- including the segment before the switch was flipped.
Recordings made before baseline removal existed carry a main switch that never
drove one, so replaying those with it applies an offset the session never had;
the standalone runner's ``--no-baseline-removal`` flag exists to suppress that.

That replay is only exact if the recorded angle is the one this class *receives*,
not the one it returns. :meth:`SensorPreprocessor.filter` never modifies the
sample it is given, so the raw angle logged by the integration stays free of the
offset. Logging the corrected angle as "raw" would make playback subtract the
offset twice.
"""

from __future__ import annotations

from loguru import logger

from hip_controller.definitions import BaselineRemovalConfig


class BaselineRemoval:
    """Capture a stand-still angle offset on trigger and subtract it thereafter."""

    def __init__(self, config: BaselineRemovalConfig) -> None:
        """Initialize the baseline removal.

        :param BaselineRemovalConfig config: Window length and stand-still
            threshold.
        :return: None
        """
        self._config: BaselineRemovalConfig = config

        self._offset_rad: float = 0.0
        self._trigger_active: bool = False
        self._collecting: bool = False
        self._samples: list[float] = []

    @property
    def offset_rad(self) -> float:
        """Angle offset currently being subtracted [rad]; 0.0 until first captured.

        :return: The active offset.
        :rtype: float
        """
        return self._offset_rad

    @property
    def is_collecting(self) -> bool:
        """Whether a baseline window is open.

        :return: True while samples are being collected.
        :rtype: bool
        """
        return self._collecting

    def set_trigger(self, active: bool) -> None:
        """Drive the baseline trigger; edges are detected internally.

        Rising edge opens a fresh window. Falling edge closes it: the mean of
        whatever still samples were gathered becomes the new offset, matching the
        Simulink release behaviour. A release with no usable sample keeps the
        previous offset rather than inventing one.

        :param bool active: Current trigger state (the main switch on the
            exosuit). Safe to call every loop iteration with an unchanged value.
        :return: None
        """
        if active and not self._trigger_active:
            self._samples.clear()
            self._collecting = True
            logger.info("Baseline window opened.")
        elif self._trigger_active and not active:
            self._close_window(released=True)

        self._trigger_active = active

    def apply(self, angle_rad: float, velocity_rad_per_sec: float) -> float:
        """Feed one sample in and return the baseline-corrected angle.

        :param float angle_rad: Raw hip angle for this sample [rad].
        :param float velocity_rad_per_sec: Raw hip velocity for this sample,
            used only to decide whether the limb is still enough to take a
            baseline [rad/s].
        :return: ``angle_rad`` minus the active offset [rad].
        :rtype: float
        """
        if self._collecting:
            self._collect(
                angle_rad=angle_rad, velocity_rad_per_sec=velocity_rad_per_sec
            )

        return angle_rad - self._offset_rad

    def reset(self) -> None:
        """Drop the offset and any open window, as after a dropout.

        :return: None
        """
        self._offset_rad = 0.0
        self._trigger_active = False
        self._collecting = False
        self._samples.clear()

    def _collect(self, angle_rad: float, velocity_rad_per_sec: float) -> None:
        """Add one sample to the open window, restarting it if the limb moved.

        :param float angle_rad: Raw hip angle for this sample [rad].
        :param float velocity_rad_per_sec: Raw hip velocity for this sample [rad/s].
        :return: None
        """
        if abs(velocity_rad_per_sec) > self._config.max_velocity_rad_per_sec:
            # Motion inside the window makes the mean meaningless. Start over
            # rather than averaging a moving leg -- this is the failure the
            # trigger exists to prevent, and it must not sneak back in when the
            # operator flips the switch mid-stride.
            self._samples.clear()
            return

        self._samples.append(angle_rad)
        if len(self._samples) >= self._config.window_sample_count:
            self._close_window(released=False)

    def _close_window(self, released: bool) -> None:
        """Turn the collected samples into the new offset and close the window.

        :param bool released: True when the window is closed by the trigger
            going low, False when it filled on its own.
        :return: None
        """
        self._collecting = False

        if not self._samples:
            if released:
                logger.warning(
                    "Baseline window closed without a still sample; "
                    f"keeping the previous offset {self._offset_rad:.4f} rad."
                )
            return

        self._offset_rad = sum(self._samples) / len(self._samples)
        logger.info(
            f"Baseline offset set to {self._offset_rad:.4f} rad "
            f"from {len(self._samples)} samples."
        )
        self._samples.clear()
