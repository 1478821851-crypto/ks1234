# Game segmentation regression fixtures

Six original screenshots supplied by the user on 2026-10-08, in attachment order (excluding the two progress screenshots). Source bytes and SHA-256 hashes are preserved.

`annotations.json` records manually checked card-background rectangles and occupied cells, not detector-generated expected output. Expected counts: A=58, B=27, C=66, D=66, E=54, F=67 (338 cards). Coordinates use exclusive right/bottom edges; tolerate small antialiasing/JPEG edge differences with IoU checks.

Coverage includes avatar frames, offline/dim cards, sparse final rows, empty + slots, message banners and changes between 5/4/3/2-column layouts. F's last card must end near y=2750, not extend through the page footer. E must retain the isolated final-row card.

The fixtures verify segmentation only; they do not assert recognized text or membership matching.
