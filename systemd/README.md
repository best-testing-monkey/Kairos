# Kairos systemd timers

User-level systemd units for Kairos automation.

## Install

```bash
# 1. Copy environment file and add Telegram credentials.
mkdir -p ~/.config/kairos
cp /media/baz/MonkeyWorks/PycharmProjects/Kairos/.env.example ~/.config/kairos/kairos.env
nano ~/.config/kairos/kairos.env

# 2. Link units into your user systemd directory.
UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$UNIT_DIR"
cp /media/baz/MonkeyWorks/PycharmProjects/Kairos/systemd/*.service /media/baz/MonkeyWorks/PycharmProjects/Kairos/systemd/*.timer "$UNIT_DIR/"

# Systemd refuses to load executable or world-writable unit files.
# (The Kairos repo is on an NTFS drive, so permissions are copied as 777.)
chmod 0644 "$UNIT_DIR"/kairos-*.service "$UNIT_DIR"/kairos-*.timer

# 3. Reload and enable timers.
systemctl --user daemon-reload
systemctl --user enable kairos-daily-signals.timer
systemctl --user enable kairos-weekly-discovery.timer
systemctl --user start kairos-daily-signals.timer
systemctl --user start kairos-weekly-discovery.timer
```

## Verify

```bash
systemctl --user list-timers
systemctl --user status kairos-daily-signals.timer
systemctl --user status kairos-weekly-discovery.timer
```

## Run manually

None of these scripts (nor `strategy/kairos_pipeline.py`) load
`~/.config/kairos/kairos.env` themselves — only the systemd units do that,
via `EnvironmentFile=`. Source it into your shell first or Telegram
notifications will silently no-op (`OpsError` caught, logged only as a
`WARNING:` line):

```bash
set -a && source ~/.config/kairos/kairos.env && set +a
```

```bash
# Daily signals (right now)
uv run /media/baz/MonkeyWorks/PycharmProjects/Kairos/scripts/kairos_daily_signals.py

# Weekly discovery (daily interval only by default)
uv run /media/baz/MonkeyWorks/PycharmProjects/Kairos/scripts/kairos_weekly_discovery.py

# Weekly discovery with hourly pass as well
uv run /media/baz/MonkeyWorks/PycharmProjects/Kairos/scripts/kairos_weekly_discovery.py --include-hourly

# finetune_next: GPU-idle check + shared GpuLock built in (also Telegram-enabled)
uv run /media/baz/MonkeyWorks/PycharmProjects/Kairos/strategy/kairos_pipeline.py --stage finetune_next
```

## Logs

```bash
journalctl --user -u kairos-daily-signals.service -f
journalctl --user -u kairos-weekly-discovery.service -f
```

Logs are also written to:

- `~/.local/state/kairos/daily_signals.log`
- `~/.local/state/kairos/weekly_discovery.log`

## Scheduling

- **Daily signals:** every day at 02:00 UTC (`kairos-daily-signals.timer`).
- **Weekly discovery:** every Sunday at 04:00 UTC (`kairos-weekly-discovery.timer`).

Edit the `OnCalendar=` lines to change times.
