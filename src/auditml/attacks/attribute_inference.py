import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from .base import AttackResult, BaseAttack


class AttributeInference(BaseAttack):
    """Simple attribute inference proxy: infer true class from model output + confidence features."""

    def run(self, member_data, nonmember_data):
        _, member_probs, member_labels = self.get_model_outputs(member_data)
        _, nonmember_probs, nonmember_labels = self.get_model_outputs(nonmember_data)

        probs = np.vstack([member_probs.numpy(), nonmember_probs.numpy()])
        labels = np.concatenate([member_labels.numpy(), nonmember_labels.numpy()])

        # Known feature = top confidence; sensitive attribute proxy = class label.
        top_conf = np.max(probs, axis=1, keepdims=True)
        X = np.hstack([probs, top_conf])
        y = labels.astype(int)

        seed = int(self.config.get("seed", 42))
        test_size = float(self.config.get("attack_test_size", 0.3))
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )

        clf = RandomForestClassifier(n_estimators=50, random_state=seed)
        clf.fit(X_train, y_train)
        pred_sensitive = clf.predict(X_test)

        # Convert to binary success/failure to keep a consistent AttackResult interface.
        success = (pred_sensitive == y_test).astype(int)
        scores = clf.predict_proba(X_test).max(axis=1)
        return AttackResult(
            predictions=success,
            ground_truth=np.ones_like(success),
            confidence_scores=scores,
            metadata={
                "mode": "class-label-proxy",
                "n_classes": int(len(np.unique(y))),
                "train_size": int(len(X_train)),
                "test_size": int(len(X_test)),
            },
        )
