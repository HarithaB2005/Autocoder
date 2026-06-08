@router.post(
    "/upload-template",
    summary="Upload a template file and get its text content",
    description=(
        "Upload your template file containing placeholders like {field_name}. "
        "Returns the text content to use as 'template_content' in /process.\n\n"
        "**Workflow:**\n"
        "1. Use /upload for your document\n"
        "2. Use /upload-template for your template\n"
        "3. Paste both into /process"
    ),
)
async def upload_template(
    file: UploadFile = File(..., description="Template file with placeholders — .txt, .md, .html etc."),
    _:    str        = Depends(require_api_key),
):
    """
    Read a template file and return its text content.
    Use the returned content as 'template_content' in /process.
    """
    MAX_SIZE = 500_000

    try:
        raw = await file.read()

        if len(raw) > MAX_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Template too large: {len(raw):,} bytes. Maximum: {MAX_SIZE:,} bytes."
            )

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw.decode("latin-1")
            except UnicodeDecodeError:
                raise HTTPException(
                    status_code=415,
                    detail=f"Template '{file.filename}' is not a readable text file."
                )

        return {
            "template_name":    file.filename,
            "template_content": text,
            "size_chars":       len(text),
            "size_bytes":       len(raw),
            "message": (
                "Copy 'template_content' and paste it into "
                "the 'template_content' field in /process."
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in /upload-template")
        raise HTTPException(status_code=500, detail=str(e))
