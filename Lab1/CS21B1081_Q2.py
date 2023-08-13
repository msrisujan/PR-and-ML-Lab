import pandas as pd

data = {
    "Name": ["Ram","Sam","Prabhu"],
    "Account Number": [9893893891 ,9893893898 , 9893893871],
    "Account Type": ["SB", "CA", "SB"],
    "Adhaar_No": [959389389173, 959389389179, 959389389159],
    "Balance": [8989839, 7690990, 989330]
}

df = pd.DataFrame(data)
df.to_csv("SBIAccountHolder.csv", index=False)

def append_record():
    name = input("Enter the name of the account holder: ")
    account_number = int(input("Enter the account number: "))
    account_type = input("Enter the account type: ")
    adhaar_number = int(input("Enter the adhaar number: "))
    balance = int(input("Enter the balance: "))
    df.loc[len(df.index)] = [name, account_number, account_type, adhaar_number, balance]
    print("Record appended successfully!")
    df.to_csv("SBIAccountHolder.csv", index=False)

def delete_record():
    account_number = int(input("Enter the account number to be deleted: "))
    if account_number not in df["Account Number"].values:
        print("Account number not found!")
    else:
        df.drop(df[df["Account Number"] == account_number].index, inplace=True)
        print("Record deleted successfully!")
        df.to_csv("SBIAccountHolder.csv", index=False)
    

def credit():
    account_number = int(input("Enter the account number: "))
    if account_number not in df["Account Number"].values:
        print("Account number not found!")
    else:
        amount = int(input("Enter the amount to be credited: "))
        df.loc[df["Account Number"] == account_number, "Balance"] += amount
        print("Amount credited successfully!")
        df.to_csv("SBIAccountHolder.csv", index=False)

def debit():
    account_number = int(input("Enter the account number: "))
    if account_number not in df["Account Number"].values:
        print("Account number not found!")
    else:
        amount = int(input("Enter the amount to be debited: "))
        if df.loc[df["Account Number"] == account_number, "Account Type"].values == "SB":
            if df.loc[df["Account Number"] == account_number, "Balance"].values - amount < 0:
                print("Insufficient balance!")
            else:
                df.loc[df["Account Number"] == account_number, "Balance"] -= amount
                print("Amount debited successfully!")
                df.to_csv("SBIAccountHolder.csv", index=False)
        else:
            df.loc[df["Account Number"] == account_number, "Balance"] -= amount
            print("Amount debited successfully!")
            df.to_csv("SBIAccountHolder.csv", index=False)
        
def print_account_details():
    account_number = int(input("Enter the account number: "))
    if account_number not in df["Account Number"].values:
        print("Account number not found!")
    else:
        print(df[df["Account Number"] == account_number])

if __name__ == "__main__":
    print("1. Append Record")
    print("2. Delete Record")
    print("3. Credit")
    print("4. Debit")
    print("5. Print Account Details")
    print("6. Exit")
    while True:
        choice = int(input("Enter your choice: "))
        if choice == 1:
            append_record()
        elif choice == 2:
            delete_record()
        elif choice == 3:
            credit()
        elif choice == 4:
            debit()
        elif choice == 5:
            print_account_details()
        elif choice == 6:
            break
        else:
            print("Invalid choice!")