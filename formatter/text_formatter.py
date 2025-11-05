class TextFormatter:
    @staticmethod
    def apply_formatting(text, entity, intent):
        if intent == "format_bold":
            return text.replace(entity, f"<b>{entity}</b>", 1)
        elif intent == "format_italic":
            return text.replace(entity, f"<i>{entity}</i>", 1)
        elif intent == "format_header":
            return f"<b><u>{entity}</u></b>\n" + text.replace(entity, "", 1)
        else:
            return text