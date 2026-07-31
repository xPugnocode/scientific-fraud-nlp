# Slideshow Source

This folder contains the OpenDocument Presentation source used to build the project slideshow.

## Files

- `slideshow.odp`: editable slideshow source for LibreOffice Impress

## Build

Open `slideshow.odp` in LibreOffice Impress to view or edit the slides.

To export the slideshow as a PDF from this folder:

```bash
libreoffice --headless --convert-to pdf slideshow.odp
```

The output PDF will be `slideshow.pdf`.
