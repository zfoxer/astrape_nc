#   Astrape for Netconf (c) 2024-2025 by Constantine Kyriakopoulos, zfox@users.sourceforge.net.

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


import os
import socket
import threading
import random
import queue
import xml.etree.ElementTree as ETree
from enum import Enum
import torch
import talos


SOCKET_FOLDER = "/tmp"
BUFFER_SIZE = 4096
HR_CTL_COMMANDS = 1
NODES = 2  # Number of devices
TRANSCEIVERS_PER_NODE = 1
SLOTS_PER_TRANSCEIVER = 49  # Number of frequency slots (transceiver EEPROM)
DEBUG = True

# For stats
totalReward = 0
totalTime = 0
currentTime = 0  # Global time


class RPC_TYPE(Enum):
    GET = "get"
    EDIT_CONFIG = "edit-config"
    UNSUPPORTED = "unsupported"


class RequestHistory:
    """
    @brief Container to store all requests along with their feedback. Talos might need 'em someday!
    """
    def __init__(self):
        self.history = []

    def add(self, pluggableId, settings):
        """
        @brief  Add a new entry to the history.
        @param  pluggableId The ID of the pluggable.
        @param  settings The settings in a tuple.
        """
        self.history.append({
            "pluggable-id": pluggableId,
            "frequency": settings.get("frequency"),
            "power": settings.get("power"),
            "feedback": settings.get("feedback", None)
        })

    def getAll(self):
        """
        @brief  Retrieve the entire history.
        """
        return self.history

    def getSize(self):
        """
        @brief  Return the number of entries in the history.
        @return  The length.
        """
        return len(self.history)

    def calculateAvgFeedback(self):
        """
        @brief  Calculate the average of feedback values.
        @return Average feedback.
        """
        feedbackValues = [entry["feedback"] for entry in self.history if entry["feedback"] is not None]
        if not feedbackValues:
            return 0.0
        return sum(feedbackValues) / len(feedbackValues)

    def dump(self):
        """
        @brief  Dump the history in a readable format.
        @return Logged history.
        """
        if not self.history:
            return "Request history is empty."
        output = ["Request History:"]
        for idx, entry in enumerate(self.history, start = 1):
            output.append(f"{idx}: {entry}")
        return "\n".join(output)

# Global request history container
requestHistory = RequestHistory()

def initML():
    """
    @brief  Initialisation of ML specifics here.
    @return agt, en Agent and environment instances
    """
    deviceCount = NODES * TRANSCEIVERS_PER_NODE
    freqSlotCount = talos.SLOTS
    model = talos.Talos(deviceCount, freqSlotCount)
    stateDict = torch.load('talos.pth', weights_only = True)
    model.load_state_dict(stateDict)
    model.eval()
    agt = talos.Agent(model, deviceCount, freqSlotCount)
    en = talos.Environment(deviceCount, freqSlotCount)
    return agt, en

agent, env = initML()

def advanceEnv(en, deviceId1, deviceId2, slot1, slot2, resultQueue):
    """
    @brief  Processes environment advancement based on events in the result queue,
            using both deviceId1 and deviceId2 for action assignments.
    """
    global currentTime, totalTime, totalReward
    if resultQueue.qsize() != 2:
        raise ValueError("Result queue must contain exactly 2 items.")

    # Retrieve config times
    (dId1, cTime1), (dId2, cTime2) = resultQueue.get(), resultQueue.get()

    # Determine min and max times
    if cTime1 < cTime2:
        minTime, minTimeId = cTime1, dId1
        maxTime, maxTimeId = cTime2, dId2
    else:
        minTime, minTimeId = cTime2, dId2
        maxTime, maxTimeId = cTime1, dId1

    # Assign the correct slots using both deviceId1 and deviceId2
    if minTimeId == deviceId1:
        action1, action2 = slot1, slot2
    elif minTimeId == deviceId2:
        action1, action2 = slot2, slot1
    else:
        raise ValueError("Unexpected device ID in result queue.")

    # Step the environment for the first device
    reward1 = en.stepRun(minTimeId, action1, currentTime, minTime)
    currentTime += minTime
    totalTime += minTime
    totalReward += reward1

    # Step the environment for the second device
    reward2 = en.stepRun(maxTimeId, action2, currentTime, maxTime - minTime)
    currentTime += maxTime - minTime
    totalTime += maxTime - minTime
    totalReward += reward2

def collectSockets(socketFolder):
    """
    @brief  Collects all the filenames related to astrape.
    @param  The folder to search in
    @return Socket filenames
    """
    try:
        allFiles = os.listdir(socketFolder)
        astrapeFiles = [file for file in allFiles if file.startswith("astrape")]
        return astrapeFiles  # Return as a list
    except FileNotFoundError:
        print("The directory does not exist or cannot be accessed.")
        return []

def generateGetRequest(pluggableId):
    """
    @brief  Generates a static get RPC request to retrieve pluggable settings.
    @param  pluggableId The ID of the pluggable, e.g., id-01-1: 01 is the node and 1 its pluggable
    @return The get RPC request as a string.
    """
    return f"""
    <rpc xmlns="urn:ietf:params:xml:ns:netconf:base:1.0" message-id="1">
        <get>
            <filter type="xpath"
                select="/pluggables/pluggable[pluggable-id='{pluggableId}']/pluggable-settings"/>
        </get>
    </rpc>
    """

def generateEditConfigRequest(pluggableId, frequency, power, configTime):
    """
    @brief  Generates an edit-config RPC request with the given frequency and power values.
    @param  pluggableId The ID of the pluggable, e.g., id-01-1: 01 is the node and 1 its pluggable.
    @param  frequency The frequency value to set.
    @param  power The power value to set.
    @param  configTime Configuration time.
    @return The edit-config RPC request as a string.
    """
    return f"""
    <rpc xmlns="urn:ietf:params:xml:ns:netconf:base:1.0" message-id="2">
        <edit-config>
            <target>
                <running/>
            </target>
            <config>
                <pluggables xmlns="urn:pluggable-config:1.0">
                    <pluggable>
                        <pluggable-id>{pluggableId}</pluggable-id>
                        <pluggable-settings>
                            <frequency>{frequency}</frequency>
                            <power>{power}</power>
                            <config_time>{configTime}</config_time>
                        </pluggable-settings>
                    </pluggable>
                </pluggables>
            </config>
        </edit-config>
    </rpc>
    """

def handleSocket(socketPath, pluggableId, rpcType, settings, resultQueue, ):
    """
    @brief  Handles the socket connecting to an agent.
    @param  pluggableId The destination ID.
    @param  rpcType The type of the RPC to send.
    @param  settings What to send.
    @param  resultQueue Store responses.
    """
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as clientSocket:
            clientSocket.connect(socketPath)
            print(f"Connected to server at {socketPath}")

            switch = {
                RPC_TYPE.GET: lambda: (
                    print(f"Sending get RPC request:\n{generateGetRequest(pluggableId)}"),
                    clientSocket.sendall(generateGetRequest(pluggableId).encode()),
                ),
                RPC_TYPE.EDIT_CONFIG: lambda: (
                    settings and print(
                        f"Sending edit-config RPC request:\n"
                        f"{edit_config_request}"
                    ),
                    clientSocket.sendall(edit_config_request.encode())
                ) if (
                    edit_config_request := generateEditConfigRequest(
                        pluggableId,
                        settings.get("frequency"),
                        settings.get("power"),
                        settings.get("config_time")
                    )
                ) else None
            }

            # Execute the corresponding action based on rpcType and handle the response
            if rpcType in switch:
                switch[rpcType]()  # Call the lambda function
                response = clientSocket.recv(BUFFER_SIZE).decode()
                root = ETree.fromstring(response) # Parse the XML
                feedback = root.attrib.get('feedback') # Get the feedback attribute
                resultQueue.put((indexFromPlugId(TRANSCEIVERS_PER_NODE, pluggableId), float(feedback)))
                updateRequestHistory(pluggableId, response, settings)
                print(f"Received RPC response:\n{response}")
            else:
                print("Invalid RPC type!")
    except Exception as ex:
        print(f"Error: {ex}")

def updateRequestHistory(pluggableId, response, settings):
    """
    @brief  Updates the request history.
    @param  pluggableId The ID to update its settings.
    @param  response Received RPC.
    @param  settings What to update locally.
    """
    if settings is not None:
        root = ETree.fromstring(response) # Parse the XML
        feedback = root.attrib.get('feedback') # Get the feedback attribute
        updatedSettings = {"frequency": settings.get("frequency"), "power": settings.get("power"),
                           "feedback": float(feedback)}
        requestHistory.add(pluggableId, updatedSettings)

def sendRpcs(pluggableId1, pluggableId2, rpcType, settings1 = None, settings2 = None):
    """
    @brief  Connect to the Netconf server and send an RPC request.
    @param  pluggableId1 First ID to send the RPC to.
    @param  pluggableId2 Second ID to send the RPC to.
    @param  settings1 Settings for the first ID.
    @param  settings2 Settings for the second ID.
    """
    print(f"Sending RPC to IDs: {pluggableId1} and {pluggableId2} at time: {currentTime:.2f}")
    try:
        # Get list of filenames in the directory
        socketPaths = collectSockets(SOCKET_FOLDER)
        threads = []
        resultQueue = queue.Queue()
        for socketPath in socketPaths:
            # Ensure the pluggable ID belongs in this socket connection
            if pluggableId1.split('-')[1] in socketPath:
                # Launch a thread for each file
                socketPath = os.path.join(SOCKET_FOLDER, socketPath)
                thread = threading.Thread(target = handleSocket,
                                          args = (socketPath, pluggableId1, rpcType, settings1, resultQueue,))
                threads.append(thread)
                thread.start()
            if pluggableId2.split('-')[1] in socketPath:
                # Launch a thread for each file
                socketPath = os.path.join(SOCKET_FOLDER, socketPath)
                thread = threading.Thread(target = handleSocket,
                                          args = (socketPath, pluggableId2, rpcType, settings2, resultQueue,))
                threads.append(thread)
                thread.start()
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        print(f"All threads have completed for pluggable IDs: {pluggableId1} and {pluggableId2}.")
        # Advance the environment
        if resultQueue.qsize() == 2:
            deviceId1 = indexFromPlugId(TRANSCEIVERS_PER_NODE, pluggableId1)
            deviceId2 = indexFromPlugId(TRANSCEIVERS_PER_NODE, pluggableId2)
            slot1 = talos.slotByFreq(int(settings1.get("frequency")))
            slot2 = talos.slotByFreq(int(settings2.get("frequency")))
            advanceEnv(env, deviceId1, deviceId2, slot1, slot2, resultQueue)

    except FileNotFoundError:
        print("The directory does not exist or cannot be accessed.")

def generatePluggablePairs(reqs):
    """
    @brief  Generates two pluggable IDs to send stuff to.
    @param  reqs Number of requests to generate.
    @return ids Generated endpoints.
    """
    ids = []
    for _ in range(reqs):
        transceiverSrcId = random.randint(1, TRANSCEIVERS_PER_NODE)
        possibleIds = [i for i in range(1, TRANSCEIVERS_PER_NODE + 1)]
        transceiverDstId = random.choice(possibleIds)
        ids.append((transceiverSrcId, transceiverDstId))
    return ids

def generateSocketPairs(reqs):
    """
    @brief  Generates two sockets to send stuff to.
    @param  reqs Number of requests.
    @return sockets Generated socket names.
    """
    if NODES <= 1:
        raise ValueError("NODES must be greater than 1 to generate valid socket pairs.")

    sockets = []
    for _ in range(reqs):
        socketSrcId = random.randint(1, NODES)
        possibleIds = [i for i in range(1, NODES + 1) if i != socketSrcId]
        socketDstId = random.choice(possibleIds)
        # Format source and destination IDs as strings with leading zero if needed
        formattedPair = (f"{socketSrcId:02}", f"{socketDstId:02}")
        sockets.append(formattedPair)
    return sockets

def learnAgentFreq(agt, en, deviceId):
    """
    @brief  Returns an efficient Q-Learned slot.
    @param  agt The ML agent.
    @param  en The ML environment.
    @param  deviceId Pluggable ID.
    @return The efficient frequency to use.
    """
    global currentTime
    unavailableSlots = en.getUnavailableSlots(deviceId, currentTime)
    usageVector = [1 if i in unavailableSlots else 0 for i in range(en.freqSlotCount)]
    action = agt.selectAction(deviceId, usageVector, test = True)
    return talos.freqBySlot(action)

def indexFromPlugId(transceiversPerNode, idStr):
    """
    @brief  Returns the index of this pluggable ID.
    @param  transceiversPerNode Transceivers per node.
    @param  idStr Pluggable ID in a string form.
    @return The index.
    """
    # Extract socket and pluggable numbers from the ID
    try:
        parts = idStr.split('-')
        if len(parts) != 3:
            raise ValueError("Invalid ID format")

        sck = int(parts[1])  # Socket number
        pluggable = int(parts[2])  # Pluggable number

        # Calculate the index
        place = (sck - 1) * transceiversPerNode + pluggable
        return place - 1
    except Exception as ex:
        return f"Error: {ex}"

def generateHrCTLRequest(reqs, continuity = False):
    """
    @brief  Simulates incoming commands from the Hierarchical Controller (HrCTL).
    @param  reqs Transceivers per node.
    @param  continuity If using same pair of frequencies (spectrum continuity constraint).
    @return Generated requests.
    """
    # Generate pluggable and socket pairs
    ids = generatePluggablePairs(reqs)
    sockets = generateSocketPairs(reqs)
    requests = []
    # Make sure at least two whiteboxes exist
    if len(ids) < 1 or len(sockets) < 1:
        return requests

    for(transceiverSrcId, transceiverDstId), (socketSrcId, socketDstId) in zip(ids, sockets):
        # Create pluggable IDs
        pluggableId1 = f"id-{socketSrcId}-{transceiverSrcId}"
        pluggableId2 = f"id-{socketDstId}-{transceiverDstId}"
        idSrc = indexFromPlugId(TRANSCEIVERS_PER_NODE, pluggableId1)
        idDst = indexFromPlugId(TRANSCEIVERS_PER_NODE, pluggableId2)
        # Get frequencies from the ol' good friend Talos
        if continuity is False:
            freqSrc = learnAgentFreq(agent, env, idSrc)
            freqDst = learnAgentFreq(agent, env, idDst)
            settingsSrc = {
                "frequency": freqSrc,
                "power": 0,
                "config_time": 0.0
            }
            settingsDst = {
                "frequency": freqDst,
                "power": 0,
                "config_time": 0.0
            }
            # Append the pair of IDs and settings to the list
            requests.append((pluggableId1, settingsSrc, pluggableId2, settingsDst))
        else:
            freq = learnAgentFreq(agent, env, idSrc)
            settings = {
                "frequency": freq,
                "power": 0,
                "config_time": 0.0
            }
            requests.append((pluggableId1, settings, pluggableId2, settings))
    return requests

def main():
    if collectSockets(SOCKET_FOLDER):
        requests = generateHrCTLRequest(HR_CTL_COMMANDS, False)
        for request in requests:
            pluggableIdSrc, settingsSrc, pluggableIdDst, settingsDst = request
            sendRpcs(pluggableIdSrc, pluggableIdDst, RPC_TYPE.EDIT_CONFIG, settingsSrc, settingsDst)

    print(f"{requestHistory.dump()}")
    print(f"End time: {currentTime}")
    print(f"Average feedback: {requestHistory.calculateAvgFeedback()}")

if __name__ == "__main__":
    main()