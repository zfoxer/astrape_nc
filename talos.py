#   Astrape for Netconf (c) 2024-2025 by Constantine Kyriakopoulos, zfox@users.sourceforge.net.
#   Talos is a bronze, let's say... robot, that guarded the ancient Crete, and it is created by God Hephaestus.
#   Or else... A neural network that decides the appropriate laser frequency to use for whitebox node
#   pluggable transceivers.

#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 2 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random
import os

MIN_FREQ_SLOT = 191300
MAX_FREQ_SLOT = 196101
FREQ_STEP = 100
PLUGGABLES = 2
SLOTS = np.array([slot for slot in range(MIN_FREQ_SLOT, MAX_FREQ_SLOT, FREQ_STEP)]).size
CSV_FILE = os.getcwd() + "/" + "frequency_dataset.csv"
DEBUG = True

def freqBySlot(index, minFreqSlot = MIN_FREQ_SLOT, maxFreqSlot = MAX_FREQ_SLOT, freqStep = FREQ_STEP):
    """
    @brief Returns the actual frequency value from its slot number.
    @param index The index of the frequency slot.
    @param minFreqSlot Left bound of slots.
    @param maxFreqSlot Right bound of slots.
    @param freqStep Distance between slots.
    @return The frequency by its index.
    """
    frequencies = np.arange(minFreqSlot, maxFreqSlot, freqStep)
    if index < 0 or index >= len(frequencies):
        raise IndexError(f"Index {index} is out of bounds for frequency slots.")
    return int(frequencies[index])

def slotByFreq(frequency):
    """
    @brief Returns the index of the frequency.
    @param frequency Transceiver frequency.
    @return The slot (index) by the given frequency.
    """
    frequencies = np.arange(MIN_FREQ_SLOT, MAX_FREQ_SLOT, FREQ_STEP)
    indices = np.where(frequencies == frequency)[0]
    if indices.size == 0:
        raise ValueError(f"Frequency {frequency} not found in the allowed range.")
    return indices[0]

class FreqConfigDataset(Dataset):
    """
    @brief Dataset Loader.
    """
    def __init__(self, csvFile):
        self.data = pd.read_csv(csvFile)
        self.freqSlotCount = SLOTS # From firmware
        self.validSlots = [slot for slot in range(MIN_FREQ_SLOT, MAX_FREQ_SLOT, FREQ_STEP)]
        self.slotToIndex = {slot: i for i, slot in enumerate(self.validSlots)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        slot = int(row['Frequency Slot'])
        if slot not in self.slotToIndex:
            raise ValueError(f"Slot {slot} is not a valid slot.")
        normalizedSlot = self.slotToIndex[slot]  # Map slot to index
        configTime = row['Configuration Time (s)']
        usageVector = np.zeros(self.freqSlotCount)
        usageVector[normalizedSlot] = 1  # Simulate slot being in use
        return normalizedSlot, configTime, usageVector

    def getTimeByFrequency(self, frequency):
        """
        @brief  Takes a frequency value as input and returns the corresponding time.
                If multiple entries exist, returns one of them randomly (uniform distro).
        @param  frequency The frequency to get its configuration time.
        @return The config time.
        """
        matching_rows = self.data[self.data['Frequency Slot'] == frequency]
        if matching_rows.empty:
            raise ValueError(f"No entries found for frequency: {frequency}")
        return random.choice(matching_rows['Configuration Time (s)'].tolist())

dataset = FreqConfigDataset(CSV_FILE)

class Talos(nn.Module):
    """
    @brief Policy network, Feedforward neural network (FNN).
    """
    def __init__(self, deviceCount, freqSlotCount, embeddingDim = 16, hiddenDim = 64):
        super(Talos, self).__init__()
        self.deviceEmbedding = nn.Embedding(deviceCount, embeddingDim)
        self.fc = nn.Sequential(
            nn.Linear(embeddingDim + freqSlotCount, hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, freqSlotCount)  # Q-Values for all frequency slots
        )

    def forward(self, deviceId, usageVector):
        deviceEmb = self.deviceEmbedding(deviceId)  # Device embedding
        x = torch.cat((deviceEmb, usageVector), dim = -1)  # Concatenate embeddings and usage vector
        return self.fc(x)  # Output Q-Values

class Agent:
    """
    @brief Config Agent.
    """
    def __init__(self, model, deviceCount, freqSlotCount, lr = 0.001, gamma = 0.99, epsilon = 0.1):
        self.deviceCount = deviceCount
        self.freqSlotCount = freqSlotCount
        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon = epsilon  # Exploration rate
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr = lr)
        self.criterion = nn.MSELoss()  # Loss function

    def selectAction(self, deviceId, usageVector, test = False):
        """
        @brief  ε-Greedy action selection.
        @param  deviceId The ID of the transceiver.
        @param  usageVector Current usage state of transceivers.
        @param  test If exploring (false) or exploiting (true).
        @return The index of the action.
        """
        with torch.no_grad():
            deviceIdTensor = torch.tensor([deviceId], dtype = torch.long)
            usageTensor = torch.tensor([usageVector], dtype = torch.float)
            qValues = self.model(deviceIdTensor, usageTensor).squeeze() # Remove 1D entries

        # Mask unavailable slots
        mask = torch.tensor(usageVector)  # 1 for unavailable, 0 for available
        qValues = qValues.masked_fill(mask > 0, float('-inf'))  # Set unavailable slots to -inf

        # 'Explore' avoids starvation
        if not test and random.random() < self.epsilon:  # Explore
            availableIndices = torch.where(mask == 0)[0].tolist()
            return random.choice(availableIndices)
        else:  # 'Exploit'
            return torch.argmax(qValues).item()

    def train(self, deviceId, usageVector, action, reward, nextDeviceId = None, nextUsageVector = None):
        """
        @brief  Updates Q-Values based on observed rewards.
        @param  deviceId The ID of the transceiver.
        @param  usageVector Current usage state of transceivers.
        @param  action Frequency index.
        @param  reward The negative configuration time.
        @param  nextDeviceId Next device ID.
        @param  nextUsageVector Next usage vector.
        """
        deviceIdTensor = torch.tensor([deviceId], dtype = torch.long)
        usageTensor = torch.tensor([usageVector], dtype = torch.float)
        qValues = self.model(deviceIdTensor, usageTensor)
        qValue = qValues[0, action]  # Extract the Q-value for the selected action

        # The next device ID is not correlated to the previous, so not used by default (None default value)
        # As reward should be the negative configuration time
        if nextDeviceId is not None:
            nextDeviceIdTensor = torch.tensor([nextDeviceId], dtype = torch.long)
            nextUsageTensor = torch.tensor([nextUsageVector], dtype = torch.float)
            nextQValues = self.model(nextDeviceIdTensor, nextUsageTensor)
            maxNextQValue = torch.max(nextQValues).item()
            target = reward + self.gamma * maxNextQValue
        else:
            target = reward

        # Ensure both tensors are 1D
        qValue = qValue.unsqueeze(0)  # Make qValue 1D
        targetTensor = torch.tensor([target], dtype = torch.float)  # Ensure target is 1D

        # Compute loss and update policy network
        loss = self.criterion(qValue, targetTensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Environment:
    """
    @brief  Config environment. A usage vector defines the frequency slot availability.
            Slot occupation time avoids frequency reusal since the request lasts for config + occupation time.
    """
    def __init__(self, deviceCount, freqSlotCount, occupationMultiplier = 0):
        self.deviceCount = deviceCount
        self.freqSlotCount = freqSlotCount
        self.occupationMultiplier = occupationMultiplier
        self.slotAvailability = {device: [0] * freqSlotCount for device in range(deviceCount)}
        self.currentData = {}  # Store configuration times for (device, freq slot)

    def stepRun(self, deviceId, action, currentTime, configTime):
        """
        @brief  Simulate slot usage and calculate reward.
        @param  deviceId The ID of the transceiver.
        @param  action Frequency index.
        @param  currentTime Current timestamp.
        @param  configTime Laser set time.
        @return reward Environmental feedback.
        """
        if (deviceId, action) not in self.currentData:
            self.currentData[(deviceId, action)] = configTime
        occupationTime = configTime + configTime * self.occupationMultiplier
        self.slotAvailability[deviceId][action] = currentTime + occupationTime
        # Higher configuration time should produce lower reward
        reward = -configTime
        return reward

    def step(self, deviceId, action, currentTime):
        """
        @brief  Simulate slot usage and get reward from dataset.
        @param  deviceId The ID of the transceiver.
        @param  action Frequency index.
        @param  currentTime Current timestamp.
        @return configTime, reward Laser set time and environmental feedback.
        """
        if (deviceId, action) not in self.currentData:
            # Get the config time from dataset
            frequency = freqBySlot(action)
            configTime = dataset.getTimeByFrequency(frequency)
            # Logic here can improve
            self.currentData[(deviceId, action)] = configTime
        else:
            configTime = self.currentData[(deviceId, action)]

        occupationTime = configTime + configTime * self.occupationMultiplier
        self.slotAvailability[deviceId][action] = currentTime + occupationTime

        # Higher configuration time should produce lower reward
        reward = -configTime
        return configTime, reward

    def getUnavailableSlots(self, deviceId, currentTime):
        """
        @brief  Return unavailable slots based on current time
        @param  deviceId The ID of the transceiver.
        @param  currentTime Current timestamp.
        @return Array with slot availability
        """
        # TODO: FIX this, availability should be per pluggable, not per slot
        return [i for i, availableTime in enumerate(self.slotAvailability[deviceId]) if currentTime < availableTime]

def trainAgent(agt, en, numEpisodes = 100):
    """
    @brief  Train this babe.
    @param  agt The agent.
    @param  en Current environment instance.
    @param  numEpisodes Training episodes.
    @return avgT, avgR Average config time and reward.
    """
    totalReward = 0
    totalTime = 0
    currentTime = 0  # Simulated global time

    for episode in range(numEpisodes):
        deviceId = random.randint(0, en.deviceCount - 1)
        unavailableSlots = en.getUnavailableSlots(deviceId, currentTime)
        usageVector = [1 if i in unavailableSlots else 0 for i in range(en.freqSlotCount)]

        action = agt.selectAction(deviceId, usageVector)
        configTime, reward = en.step(deviceId, action, currentTime)

        currentTime += configTime  # Increment global time
        totalTime += configTime
        totalReward += reward
        agt.train(deviceId, usageVector, action, reward)

        if episode % 100 == 0:
            print(f"Episode {episode}, Device {deviceId}, Slot {action}, Config Time {configTime:.2f}, Time {currentTime:.2f}")

    avgT = totalTime / numEpisodes
    avgR = totalReward / numEpisodes
    print(f"Average Config Time: {avgT:.2f}, Average Reward: {avgR:.2f}")
    return avgT, avgR

def testAgent(agt, en, numTests = 100):
    """
    @brief  Test this babe.
    @param  agt The agent.
    @param  en Current environment instance.
    @param  numTests Testing iterations.
    @return avgT, avgR Average config time and reward.
    """
    totalReward = 0
    totalTime = 0
    currentTime = 0  # Simulated global time

    for test in range(numTests):
        deviceId = random.randint(0, en.deviceCount - 1)
        unavailableSlots = en.getUnavailableSlots(deviceId, currentTime)
        usageVector = [1 if i in unavailableSlots else 0 for i in range(en.freqSlotCount)]

        action = agt.selectAction(deviceId, usageVector, test = True)
        configTime, reward = en.step(deviceId, action, currentTime)

        currentTime += configTime  # Increment global time
        totalTime += configTime
        totalReward += reward

        if test % 100 == 0:
            print(f"Test: Device {deviceId}, Slot {action}, Config Time {configTime:.2f}, Time {currentTime:.2f}")

    avgT = totalTime / numTests
    avgR = totalReward / numTests
    print(f"Average Config Time: {avgT:.2f}, Average Reward: {avgR:.2f}")
    return avgT, avgR

def main():
    deviceCount = PLUGGABLES
    freqSlotCount = SLOTS
    model = Talos(deviceCount, freqSlotCount)
    #   Load previously saved model
    #stateDict = torch.load('talos.pth', weights_only = True)
    #model.load_state_dict(stateDict)
    #model.eval()
    agent = Agent(model, deviceCount, freqSlotCount)
    env = Environment(deviceCount, freqSlotCount)

    print("Training...")
    avgTime, avgReward = trainAgent(agent, env, numEpisodes = 1001)
    #   Save serialised model
    torch.save(model.state_dict(), 'talos.pth')

    print("\nTesting...")
    avgTime, avgReward = testAgent(agent, env, numTests = 1000)

if __name__ == "__main__":
    main()