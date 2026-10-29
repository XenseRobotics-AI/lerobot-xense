"""Which session failures still take the configured B601 return to zero on exit.

Kept free of the native SDK so CLI teardown can import it on any failure path.
"""


class B601ControlFaultError(RuntimeError):
    """The native controller reported a fault; the arm must not be commanded further."""


def exit_reset_allowed(error: BaseException | None) -> bool:
    """Whether disconnect() should still apply reset_on_exit after ``error``.

    Disabling the motors in place drops a B601 held in MIT mode with gravity
    feed-forward, so only an arm/controller fault justifies skipping the
    return: a fault raised by the driver, or rebot_rt refusing a command with
    "control fault:". Everything else -- normal exit, Ctrl+C, a teleop/VR,
    camera, gripper or IK/pose-input error -- leaves the arm controller
    healthy and still returns. disconnect() itself re-checks the controller
    before moving and skips the return on a fault or a lost controller.
    """
    if isinstance(error, B601ControlFaultError):
        return False
    return not (isinstance(error, RuntimeError) and str(error).startswith("control fault:"))
