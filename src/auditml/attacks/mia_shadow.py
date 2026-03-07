import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .base import AttackResult, BaseAttack


class ShadowMIA(BaseAttack):
    """A lightweight shadow-style MIA approximation using confidence+entropy features."""

    @staticmethod
    def _features_from_probs(probs):
        max_prob = np.max(probs, axis=1)
        entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
        return np.stack([max_prob, entropy], axis=1)

    def run(self, member_data, nonmember_data):
        _, member_probs, _ = self.get_model_outputs(member_data)
        _, nonmember_probs, _ = self.get_model_outputs(nonmember_data)

        m = member_probs.numpy()
        n = nonmember_probs.numpy()

        X = np.vstack([self._features_from_probs(m), self._features_from_probs(n)])
        y = np.concatenate([np.ones(len(m), dtype=int), np.zeros(len(n), dtype=int)])

        test_size = float(self.config.get("attack_test_size", 0.3))
        seed = int(self.config.get("seed", 42))
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )

        clf = LogisticRegression(max_iter=500, random_state=seed)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        return AttackResult(
            predictions=pred,
            ground_truth=y_test,
            confidence_scores=proba,
            metadata={
                "feature_names": ["max_prob", "entropy"],
                "classifier": "logistic_regression",
                "train_size": int(len(X_train)),
                "test_size": int(len(X_test)),
            },
        )
