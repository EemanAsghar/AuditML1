from .base import BaseAttack


class AttributeInference(BaseAttack):
    def run(self, member_data, nonmember_data):
        return {"message": "Attribute inference scaffold ready; plug in tabular/image-specific pipeline."}
