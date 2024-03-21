import re
import torch
import torch.nn as nn

log = []

# Generate log for 16 weeks with specified patterns
for week in range(1, 33):
    for day in range(1, 8):
        if week % 2 == 0 and week % 3 != 0:  # Even weeks excluding multiples of 3
            if day in [1, 2, 3, 4, 5]:  # Monday to Friday
                wake_time = "6:{:02d}".format(torch.randint(0, 16, (1,)).item())  # Random wake time between 6:00 and 6:30
                sleep_time = "22:{:02d}".format(torch.randint(45, 60, (1,)).item())  # Random sleep time between 22:30 and 23:00
            else:
                wake_time = "9:{:02d}".format(torch.randint(45, 60, (1,)).item())  # Random wake time between 9:30 and 10:00
                sleep_time = "23:{:02d}".format(torch.randint(45, 60, (1,)).item())  # Random sleep time between 23:30 and 23:59
        elif week % 3 == 0:  # Every third week
            if day == 2:  # Tuesday
                wake_time = "5:30"
            elif day == 6:  # Saturday
                wake_time = "11:00"
            else:
                wake_time = "7:{:02d}".format(torch.randint(0, 16, (1,)).item())  # Random wake time between 7:00 and 7:15
            sleep_time = "22:{:02d}".format(torch.randint(30, 45, (1,)).item())  # Random sleep time between 22:30 and 23:00
        else:  # Odd weeks
            if day in [1, 2, 3, 4, 5]:  # Monday to Friday
                wake_time = "7:{:02d}".format(torch.randint(0, 16, (1,)).item())  # Random wake time between 7:00 and 7:15
                sleep_time = "22:{:02d}".format(torch.randint(45, 60, (1,)).item())  # Random sleep time between 22:30 and 23:00
            else:
                wake_time = "9:{:02d}".format(torch.randint(45, 60, (1,)).item())  # Random wake time between 9:30 and 10:00
                sleep_time = "23:{:02d}".format(torch.randint(45, 60, (1,)).item())  # Random sleep time between 23:30 and 23:59
        log.append("Week {}, Day: {}, Woke: {}, Slept: {}".format(week, day, wake_time, sleep_time))

# Print the modified log
for entry in log:
    print(entry)


# Define a function to extract sleep schedule
def extract_sleep_schedule(log):
    # Regular expression pattern to extract week number, day of the week, wake time, and sleep time
    pattern = r"Week (\d+), Day: (\w+), Woke: (\d+):(\d+), Slept: (\d+):(\d+)"

    # Initialize list to store extracted information
    sleep_schedule = []

    # Iterate through each log entry
    for entry in log:
        # Extract information using regular expression
       # Extract information using regular expression
        match = re.match(pattern, entry)

        if match:
            week_number = int(match.group(1))
            day_number = int(match.group(2))  # Extract day number as an integer
            wake_hour = int(match.group(3))
            wake_minute = int(match.group(4))
            sleep_hour = int(match.group(5))
            sleep_minute = int(match.group(6))

            # Convert wake and sleep times to minutes since midnight
            wake_time = wake_hour * 60 + wake_minute
            sleep_time = sleep_hour * 60 + sleep_minute

            # Determine if it's the second week and third week
            is_second_week = 1 if week_number % 2 == 0 else 0
            is_third_week = 1 if week_number % 3 == 0 else 0

            # Append extracted information to sleep schedule
            sleep_schedule.append([week_number, day_number, is_second_week, is_third_week, wake_time, sleep_time])


    return sleep_schedule

# Extract sleep schedule from log entries
sleep_schedule = extract_sleep_schedule(log)

# Define a simple neural network architecture
class SleepSchedulePredictor(nn.Module):
    def __init__(self):
        super(SleepSchedulePredictor, self).__init__()
        self.fc1 = nn.Linear(4, 64)  # Input size: 4 (week number, day number, is_second_week, is_third_week)
        self.fc2 = nn.Linear(64, 2)  # Output size: 2 (wake time, sleep time)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation
        x = self.fc2(x)
        return x

# Convert sleep schedule data into PyTorch tensors
sleep_schedule_tensor = torch.tensor(sleep_schedule, dtype=torch.float32)

# Extract features (week number, day number, is_second_week, is_third_week)
X_train = sleep_schedule_tensor[:, :4]

# Extract targets (wake time, sleep time)
y_train = sleep_schedule_tensor[:, 4:]
# Define a function to clip the time within the valid range (00:00 to 23:59)


# Normalize input features (week number, day number, is_second_week, is_third_week)
X_train_normalized = (X_train - X_train.mean(dim=0)) / X_train.std(dim=0)

# Create an instance of the sleep schedule predictor neural network
model = SleepSchedulePredictor()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.010)
print(f'\n')
# Train the neural network
num_epochs = 50000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_normalized)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if epoch >= 7000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.005
    if epoch >= 22000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.003
    # Update learning rate
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = 10000 / (2000 + epoch*5)
        
    
    # if (epoch+1) % 1000 == 0:
    #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    if (epoch+1) % 3000 == 0:
        percentage = (epoch+1)/num_epochs * 100
        print(f'{percentage:.0f}%')

start_week = sleep_schedule[0][0]
end_week = sleep_schedule[-1][0]
print(f"\n\nThe log covers Week {start_week} to Week {end_week}.\n")
# Example usage for prediction
# Week number and day number for prediction

def get_day_name(day_number):
    day_names = {
        1: "Monday",
        2: "Tuesday",
        3: "Wednesday",
        4: "Thursday",
        5: "Friday",
        6: "Saturday",
        7: "Sunday"
    }
    return day_names.get(day_number, "Invalid day")

def predict_sleep_schedule():
    while True:
        # Week number and day number for prediction
        week_number = int(input("\nEnter a week in the future (Enter 0 to exit the program): "))
        if week_number == 0:
            print("\nExiting the program.")
            break
        elif week_number == -1:
           predict_sleep_schedule2() 
           continue
        elif week_number < 1:
            print("\nInvalid week number. Please enter a number greater than or equal to 1.")
            continue
        
        day_number = int(input("\nEnter a day of the week (1 for Monday, 7 for Sunday): "))
        if day_number < 1 or day_number > 7:
            print("\nInvalid day number. Please enter a number between 1 and 7.")
            continue

        # Determine if it's the second week and third week
        is_second_week = 1 if week_number % 2 == 0 else 0
        is_third_week = 1 if week_number % 3 == 0 else 0

        # Normalize input features
        input_features = torch.tensor([[week_number, day_number, is_second_week, is_third_week]], dtype=torch.float32)
        input_features_normalized = (input_features - X_train.mean(dim=0)) / X_train.std(dim=0)

        predicted_times = model(input_features_normalized)

        # Convert predicted times back to minutes since midnight
        predicted_wake_time = predicted_times[0][0].item()
        predicted_sleep_time = predicted_times[0][1].item()

        # Round predicted times to the nearest 10-minute interval
        rounded_predicted_wake_time = round(predicted_wake_time / 5) * 5
        rounded_predicted_sleep_time = round(predicted_sleep_time / 5) * 5

        rounded_predicted_sleep_time = min(rounded_predicted_sleep_time, 1435)  # 1439 minutes = 23:59

        # Print predicted wake and sleep times
        print(f'\n\nWeek {week_number}, {get_day_name(day_number)}:')
        print(f'Predicted wake time: {int(rounded_predicted_wake_time / 60)}:{int(rounded_predicted_wake_time % 60):02d}')
        print(f'Predicted sleep time: {int(rounded_predicted_sleep_time / 60)}:{int(rounded_predicted_sleep_time % 60):02d}\n')

def predict_sleep_schedule2():
    # Get the last recorded week and day
    last_entry = log[-1]
    last_week_number = int(re.match(r"Week (\d+)", last_entry).group(1))
    last_day_number = int(re.match(r".*Day: (\d+)", last_entry).group(1))

    # Calculate the day after the last recorded day
    next_week_number = last_week_number
    next_day_number = last_day_number + 1
    if next_day_number > 7:
        next_day_number = 1
        next_week_number += 1

    # Predict and print the sleep schedule for the next 14 days
    for _ in range(14):
        is_second_week = 1 if next_week_number % 2 == 0 else 0
        is_third_week = 1 if next_week_number % 3 == 0 else 0

        # Normalize input features
        input_features = torch.tensor([[next_week_number, next_day_number, is_second_week, is_third_week]], dtype=torch.float32)
        input_features_normalized = (input_features - X_train.mean(dim=0)) / X_train.std(dim=0)

        predicted_times = model(input_features_normalized)

        # Convert predicted times back to minutes since midnight
        predicted_wake_time = predicted_times[0][0].item()
        predicted_sleep_time = predicted_times[0][1].item()

        # Round predicted times to the nearest 10-minute interval
        rounded_predicted_wake_time = round(predicted_wake_time / 5) * 5
        rounded_predicted_sleep_time = round(predicted_sleep_time / 5) * 5

        rounded_predicted_sleep_time = min(rounded_predicted_sleep_time, 1435)  # 1439 minutes = 23:59

        # Print predicted wake and sleep times
        print(f'\nWeek {next_week_number}, Day {next_day_number}:')
        print(f'Predicted wake time: {int(rounded_predicted_wake_time / 60)}:{int(rounded_predicted_wake_time % 60):02d}')
        print(f'Predicted sleep time: {int(rounded_predicted_sleep_time / 60)}:{int(rounded_predicted_sleep_time % 60):02d}')

        # Update next_week_number and next_day_number for the next prediction
        next_day_number += 1
        if next_day_number > 7:
            next_day_number = 1
            next_week_number += 1


# Call the function to start predicting
choice = int(input("\nChoose mode (1 or 2). Any other entry will exit the program: "))        
if choice == 2:
    predict_sleep_schedule()
elif choice == 1:
    predict_sleep_schedule2()
