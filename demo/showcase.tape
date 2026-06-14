# mic-compare showcase: on-device two-tier (Nemotron ANE + Voxtral GPU) vs cloud DeepGram
# Records the live column HUD; model-load is hidden so the GIF is just the transcription.
Output showcase.gif
Output showcase.mp4

Set Shell "zsh"
Set FontSize 15
Set Width 1500
Set Height 760
Set Padding 18

Hide
Type "cd /Volumes/DATA/mlx-audio-swift-worktrees/nemotron-session && B=.build/arm64-apple-macosx/release/mic-compare"
Enter
Type "$B --language en --wav demo/clip22.wav --only two-tier-ane,deepgram --cloud --dg-model nova-3"
Enter
Sleep 14s
Show
Sleep 26s
