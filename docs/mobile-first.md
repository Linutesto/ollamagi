# Mobile-First Workflow

OllamAGI's dashboard is designed to be fully usable from a phone browser. No separate mobile app — just a responsive web UI that works over an SSH tunnel.

## Setup (Termux on Android)

Install Termux from F-Droid, then:

```bash
pkg install openssh
ssh -L 7654:localhost:7654 youruser@yourserver -N
```

Open `http://localhost:7654` in your mobile browser (Chrome/Firefox both work).

## Dashboard Tabs

### Flows
Lists all flows (running, finished, failed, stopped). Tap any flow to open its detail view.

**Flow detail shows:**
- Status badge with live animation for running flows
- Token usage (total · prompt in · completion out · call count)
- Stop button and Steer panel (running flows only)
- Task/subtask tree with collapsible cards
- Subtask results and retry badges
- Replan banner when the flow adapted

Cards stay open during auto-refresh (every 4 seconds while running).

### Run
Quick launch presets with per-type configuration:
- **Build an Agent** — agent type, language, deployment, extras
- **Product / ROI** — niche, timeline, revenue model, budget
- **Research** — topic, depth, output format
- **Security** — target, test type, report style

Each preset builds a natural-language objective and launches the flow.

### Terminal
Live terminal tab:
- Select a flow to exec commands in its container
- Or exec on the host directly
- Color-coded output: stdout (green) · stderr (red) · command (blue) · system (grey)

### Logs
Live log stream from all flows. Filter by level: all · info · warn · error.
Badge shows unread count.

### Memory
Search cognitive memory beliefs. Shows active goals at the top.
Manual belief entry supported.

### System
- Hardware info (configured in `.env`)
- Service health: Ollama · Memory · WebSocket
- Token usage: session · all-time · per-flow breakdown
- Session reset button

## Flow Control

**Stop:** Press "⬛ Stop Flow" — flow halts within ~500ms.

**Steer:** Type a mid-flow instruction (e.g. "focus on Python only, skip the research phase") and press Send. The agents will replan the remaining tasks to match.

## SSH Tunnel Tips

- Run the SSH command in Termux background: `ssh -L 7654:localhost:7654 user@host -N &`
- Use Termux:Widget or a shortcut app to one-tap the tunnel
- The dashboard auto-reconnects WebSocket on connection drop
- Tailscale works great as an alternative to port-forwarding
