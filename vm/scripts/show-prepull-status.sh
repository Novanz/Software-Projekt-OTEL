#!/usr/bin/env bash

STATUS_FILE=/run/podman-prepull.status
STATE_FILE=/run/podman-prepull.state

is_interactive_ssh() {
    [ -n "${SSH_CONNECTION:-}" ] && [ -t 1 ]
}

should_show() {
    [ -f "$STATUS_FILE" ] || return 1
    return 0
}

print_status() {
    echo
    echo "=== FCOS bootstrap status ==="
    cat "$STATUS_FILE"
    echo "============================="
    echo
}

if is_interactive_ssh && should_show; then
    print_status
fi
