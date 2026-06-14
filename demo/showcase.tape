# mic-compare showcase — 3 engines, ~55s. Fully auto-synced via Wait:
#   Show fires when the live HUD appears (load done), recording ends after the
#   COMPARISON panel renders (feed done) — independent of load time & feed length.
Output showcase.gif
Output showcase.mp4

Set Shell "zsh"
Set FontSize 14
Set Width 1900
Set Height 900
Set Padding 16

Hide
Type "cd /Volumes/DATA/mlx-audio-swift-worktrees/nemotron-session && B=.build/arm64-apple-macosx/release/mic-compare"
Enter
Type "$B --language en --wav demo/clip55.wav --only two-tier-ane,deepgram,gemini --cloud --dg-model nova-3"
Enter
Wait+Screen@60s /TWO-TIER/
Show
Wait+Screen@90s /COMPARISON/
Sleep 4s
