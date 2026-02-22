# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("students.csv")

# Show data
print("Student Data:\n", data)

# Calculate Total Marks
data["Total"] = data[["Math","Science","English"]].sum(axis=1)

# Calculate Average using NumPy
data["Average"] = np.mean(
    data[["Math","Science","English"]], axis=1)

print("\nUpdated Data:\n", data)

# Find topper
topper = data.loc[data["Total"].idxmax()]
print("\nTopper:\n", topper)

# -------- Visualization --------

# Bar Chart
plt.figure()
plt.bar(data["Name"], data["Total"])
plt.title("Total Marks of Students")
plt.xlabel("Students")
plt.ylabel("Total Marks")
plt.show()

# Line Graph
plt.figure()
plt.plot(data["Name"], data["Average"], marker='o')
plt.title("Average Marks")
plt.xlabel("Students")
plt.ylabel("Average")
plt.show()

# Pie Chart
plt.figure()
plt.pie(data["Total"], labels=data["Name"], autopct='%1.1f%%')
plt.title("Marks Distribution")
plt.show()