from .base import BaseAttack


class ShadowMIA(BaseAttack):
    def run(self, member_data, nonmember_data):
        raise NotImplementedError("Shadow MIA pipeline scaffold created; implement training + attack classifier workflow.")
