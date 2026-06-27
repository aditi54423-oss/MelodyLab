from datasets import load_dataset

ds = load_dataset("sander-wood/melodyhub")

count = 0

for row in ds["train"]:
    if row["task"] == "generation":
        print("FOUND GENERATION ROW")
        print("Dataset:", row["dataset"])
        print("Task:", row["task"])
        print("Input:")
        print(row["input"][:500])
        print("\nOutput:")
        print(row["output"][:1000])
        count += 1

    if count == 3:
        break

print("Shown generation examples:", count)