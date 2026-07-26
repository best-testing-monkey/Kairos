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
systemctl --user enable kairos-idle-finetune.timer
systemctl --user start kairos-daily-signals.timer
systemctl --user start kairos-weekly-discovery.timer
systemctl --user start kairos-idle-finetune.timer
```

## Verify

```bash
systemctl --user list-timers
systemctl --user status kairos-daily-signals.timer
systemctl --user status kairos-weekly-discovery.timer
systemctl --user status kairos-idle-finetune.timer
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

# Idle-GPU fine-tuning (checks GPU util and lock; exits 0 if skipped)
uv run /media/baz/MonkeyWorks/PycharmProjects/Kairos/scripts/kairos_idle_finetune.py

# Idle fine-tuning with skip notifications
uv run /media/baz/MonkeyWorks/PycharmProjects/Kairos/scripts/kairos_idle_finetune.py --notify-skip

# finetune_next directly via the pipeline (also Telegram-enabled)
uv run /media/baz/MonkeyWorks/PycharmProjects/Kairos/strategy/kairos_pipeline.py --stage finetune_next
```

## Logs

```bash
journalctl --user -u kairos-daily-signals.service -f
journalctl --user -u kairos-weekly-discovery.service -f
journalctl --user -u kairos-idle-finetune.service -f
```

Logs are also written to:

- `~/.local/state/kairos/daily_signals.log`
- `~/.local/state/kairos/weekly_discovery.log`
- `~/.local/state/kairos/idle_finetune.log`

## Scheduling

- **Daily signals:** every day at 02:00 UTC (`kairos-daily-signals.timer`).
- **Weekly discovery:** every Sunday at 04:00 UTC (`kairos-weekly-discovery.timer`).
- **Idle fine-tuning:** every 30 minutes (`kairos-idle-finetune.timer`).

Edit the `OnCalendar=` lines to change times.

## Idle fine-tuning configuration

The idle fine-tuning runner rotates through the configured symbols so the same
instrument is not retrained repeatedly. It also enforces a cooldown between
training runs for the same symbol (default 24 hours).

Useful environment variables (set them in `~/.config/kairos/kairos.env` or pass
them on the command line):

| Variable | Default | Description |
|----------|---------|-------------|
| `KAIROS_IDLE_FINETUNE_SYMBOLS` | *(all tickers in local price cache DB)* | Space/comma-separated symbols to rotate through; overrides DB lookup |
| `KAIROS_IDLE_FINETUNE_SYMBOL` | *(none)* | Single symbol (legacy; use `KAIROS_IDLE_FINETUNE_SYMBOLS` instead) |
| `KAIROS_IDLE_FINETUNE_COOLDOWN_SECONDS` | `86400` (24h) | Seconds before retraining the same symbol |
| `KAIROS_IDLE_NOTIFY_SKIP` | `0` | Send Telegram when a cycle is skipped (cooldown, busy GPU, etc.) |

State is persisted in `~/.local/state/kairos/idle_finetune_state.json` so
rotation and cooldown survive reboots.
