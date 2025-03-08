from typing import Optional


class ComponentBase:
    component: Optional[str] = None
    props: Optional[dict] = None
    content: Optional[list] = None
    slots: Optional[dict] = None
    html: Optional[str] = None
    text: Optional[str] = None
    __fields__ = ["component", "props", "content", "slots", "html", "text"]

    def __init__(
        self,
        *content,
        slots: Optional[dict] = None,
        html: Optional[str] = None,
        text: Optional[str] = None,
        **props,
    ):
        self.content = list(content) or None
        self.slot = slots
        self.html = html
        self.text = text
        self.props = props or None

    def as_dict(self):
        result = {}
        for f in self.__fields__:
            value = getattr(self, f)
            if value is None:
                continue
            if f == "content":
                value = [c.as_dict() for c in value]
            result[f] = value
        return result

    def __repr__(self):
        return str(self.as_dict())


class VForm(ComponentBase):
    component: Optional[str] = "VForm"


class VRow(ComponentBase):
    component = "VRow"


class VCol(ComponentBase):
    component = "VCol"


class VSwitch(ComponentBase):
    component = "VSwitch"


class VSelect(ComponentBase):
    component = "VSelect"


class VBtn(ComponentBase):
    component = "VBtn"


class VTextField(ComponentBase):
    component = "VTextField"


class VCronField(ComponentBase):
    component = "VCronField"


class VTabs(ComponentBase):
    component = "VTabs"


class VTab(ComponentBase):
    component = "VTab"


class VWindow(ComponentBase):
    component = "VWindow"


class VWindowItem(ComponentBase):
    component = "VWindowItem"


class VBadge(ComponentBase):
    component = "VBadge"


class VTextarea(ComponentBase):
    component = "VTextarea"


class VApexChart(ComponentBase):
    component = "VApexChart"


class VDataTableVirtual(ComponentBase):
    component = "VDataTableVirtual"
