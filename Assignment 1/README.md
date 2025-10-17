Name: Kavin Ravi
ID #: 2420447
Class: CPSC393 - Machine Learning
Assignment: Assignment 1

Known issues: ran into mostly syntax issues. No dependency issues, and as this was the first assignment I didn't really expect much difficulty. As for sources, I had to ask ChatGPT why the way I was passing in "features" worked for SVM but not Logistic Regression, to which it informed me that it was makecolumntransformer throwing the error. Since it was through the Codex extension, I can't include the link to the conversation, but the prompt was:

"feats_pca = df[[c for c in df.columns if c != "Y"]]

feats_pca = df[features]
empty pipeline
z = make_column_transformer((StandardScaler(), feats_pca),
remainder = "passthrough")

pca = PCA()

pipe_pca = Pipeline([
("z", z),
("pca", pca)
])

fit
pipe_pca.fit(feats_pca)

ValueError: No valid specification of the columns. Only a scalar, list or slice of all integers or all strings, or boolean mask is allowed"

and the response was:
"You passed a DataFrame where make_column_transformer expects a column spec. Use the column names (or skip the transformer).

Issue: feats_pca is a DataFrame; make_column_transformer((StandardScaler(), feats_pca), ...) needs a list of columns, not the DataFrame.

Simplest fix (no ColumnTransformer):

feats = df.drop(columns=["Y"])
pipe_pca = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA())
])
pipe_pca.fit(feats)
If you want to keep make_column_transformer:

feats_pca = df.drop(columns=["Y"])
z = make_column_transformer((StandardScaler(), list(feats_pca.columns)),
                            remainder="drop")  # or "passthrough" if desired
pipe_pca = Pipeline([
    ("z", z),
    ("pca", PCA())
])
pipe_pca.fit(feats_pca)
With mixed dtypes (only scale numeric):

from sklearn.compose import make_column_selector
z = make_column_transformer(
    (StandardScaler(), make_column_selector(dtype_include=np.number)),
    remainder="drop"
)
pipe_pca = Pipeline([("z", z), ("pca", PCA())]).fit(df.drop(columns=["Y"]))
Note: PCA expects only numeric features; ensure non-numeric columns aren’t passed through."

This is technically the conversation I had when I ran into the SAME issue in my PCA pipeline, but the prompt and response were identical with the logistic regression pipeline. Note that I did not copy the responses for either. 