def instantiate_template(template: str, slot_values: dict) -> str:
    # simple str.format-style templating
    return template.format(**slot_values)