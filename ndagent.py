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
import re
import signal
import socket
import xml.etree.ElementTree as ETree
from modules.pluggable_config import pluggable_config as PluggableConfig
from enum import Enum
import threading
import subprocess
import packetctl as controller

SOCKET_FOLDER = "/tmp"
BUFFER_SIZE = 4096
NAMESPACE = "urn:ietf:params:xml:ns:netconf:base:1.0"
WOPEN_TOOL = os.getcwd() + "/" + "wopen_sim" # Simulate this tool
TRANSCEIVERS = controller.TRANSCEIVERS_PER_NODE
FREQ_LOW = 191300
FREQ_HIGH = 196100
POW_LOW = -6.0
POW_HIGH = 1.0
CONFIG_TIME_MAX = 150.0
DEBUG = True

class RPC_TYPE(Enum):
    GET = "get"
    EDIT_CONFIG = "edit-config"
    UNSUPPORTED = "unsupported"

class ConfigHistory:
    """
    @brief  Container to store all edit-config requests. Might need 'em someday!
    """
    def __init__(self):
        self.history = []

    def add(self, pluggableId, settings):
        """
        @brief  Adds a new entry to the history.
        @param  pluggableId Transceiver ID to add.
        @param  settings The settings for this transceiver.
        """
        self.history.append({
            "pluggable-id": pluggableId,
            "frequency": settings.get("frequency"),
            "power": settings.get("power"),
            "config_time": settings.get("config_time", None)
        })

    def getAll(self):
        """
        @brief  Retrieves the entire history.
        @return Returns the history log.
        """
        return self.history
    
    def dump(self):
        """
        @brief  Dump the history in a readable format.
        @return The readable format of log history.
        """
        if not self.history:
            return "Config history is empty."
        output = ["Config History:"]
        for idx, entry in enumerate(self.history, start = 1):
            output.append(f"{idx}: {entry}")
        return "\n".join(output)


# Dictionary to store outputs for each thread
threadOutputs = {}

# List to keep track of threads
threads = []

# Global history container
configHistory = ConfigHistory()

exitApplication = False
socketNumber = str(00)
socketPath = ""

def initSocketPath(directory):
    """
    @brief  Creates the next available socket name in two global variables.
    """
    global socketPath
    if socketPath != "":
        return
    
    # Regex pattern to match files like from "astrape01.sock" to "astrape99.sock"
    pattern = re.compile(r'astrape(\d{2})\.sock')
    maxNumber = 0

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            number = int(match.group(1))
            if number > maxNumber:
                maxNumber = number

    # Keep the largest, plus one
    nextNumber = maxNumber + 1
    socketPath = f"astrape{nextNumber:02}.sock"
    socketPath = os.path.join(SOCKET_FOLDER, socketPath)
    global socketNumber
    socketNumber = str(f"{nextNumber:02}")

def updatePluggableHistory(pluggableId, yangInstance):
    """
    @brief  Update the pluggable settings for a given pluggableId based on the edit-config request.
            Store the updated values in the config history.
    @param  pluggableId Transceiver ID.
    @param  yangInstance The YANG model instance in memory.
    @return Boolean indication of the outcome.
    """
    try:
        # Access pluggable-settings
        pluggableSettings = yangInstance.pluggables.pluggable[pluggableId].pluggable_settings
        updatedSettings = {}

        # Handle frequency
        frequency = int(pluggableSettings.frequency)
        updatedSettings["frequency"] = frequency
        print(f"Frequency for '{pluggableId}' set to {frequency}")

        # Handle power
        power = float(pluggableSettings.power)
        updatedSettings["power"] = power
        print(f"Power for '{pluggableId}' set to {power}")

        # Handle config_time
        configTime = float(pluggableSettings.config_time)
        updatedSettings["config_time"] = configTime
        print(f"Config time for '{pluggableId}' set to {configTime}")

        # Store the updated values in the history container
        configHistory.add(pluggableId, updatedSettings)
        print("Config History:", configHistory.getAll())
        return True 
    except Exception as ex:
        print(f"Error updating pluggable settings for '{pluggableId}': {ex}")
        return False

def signalHandler(signum, frame):
    """
    @brief  Signal handler trick to stop the server.
    """
    global exitApplication
    exitApplication = True

def runShellCommand(command, inputText = None):
    """
    @brief  Function to run a shell command and capture its output.
    @param  command The shell command.
    @param  inputText Extra text if needed.
    @return stdout, stderr, returncode  Standard our/error/return_code.
    """
    try:
        # Run the command and capture its output
        process = subprocess.Popen(
            command,
            stdin = subprocess.PIPE,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            shell = True,
            text = True  # Input and output is treated as text
        )
        stdout, stderr = process.communicate(input = inputText)
        return stdout, stderr, process.returncode
    except Exception as ex:
        return None, str(ex), -1

def threadTarget(command, inputText, threadId):
    """
    @brief  Target function for the thread to execute a shell command.
    @param  command The shell command.
    @param  inputText Extra text if needed.
    @param  threadId Thread ID.
    """
    stdout, stderr, returnCode = runShellCommand(command, inputText)
    
    # Store the output in the thread_outputs dictionary
    threadOutputs[threadId] = {
        "command": command,
        "input": inputText,
        "output": stdout.strip(),
        "errors": stderr.strip(),
        "return_code": returnCode
    }

def executeCommandInThread(commandName, commandArgs):
    """
    @brief  Executes a command with arguments in a thread.
    @param commandName The base command to execute (e.g., 'wopen').
    @param commandArgs The arguments for the command (e.g., '--laser-freq 193100').
    @return threadId Thread ID.
    """
    fullCommand = f"{commandName} {commandArgs}"
    commandInput = ""  # Example input to the command
    
    # Generate a unique thread ID
    threadId = len(threads) + 1  # Unique ID based on thread count
    print(fullCommand)
    # Create and start the thread
    thread = threading.Thread(target = threadTarget, args=(fullCommand, commandInput, threadId))
    threads.append(thread)
    thread.start()
    return threadId

def waitForAllThreads():
    """
    @brief  Waits for all threads to complete.
    """
    for thread in threads:
        thread.join()

def initializeYangModel():
    """
    @brief  Initialises the YANG model.
    @return yangInstance The YANG instance.
    """
    initSocketPath(SOCKET_FOLDER)
    yangInstance = PluggableConfig()
    
    for i in range(TRANSCEIVERS):
        # Add a new pluggable to the yangInstance
        pluggableId = f"id-{socketNumber}-{i+1}"
        yangInstance.pluggables.pluggable.add(pluggableId)
        frequency = FREQ_LOW + i * 100
        power = 0
        yangInstance.pluggables.pluggable[pluggableId].pluggable_settings.frequency = frequency
        yangInstance.pluggables.pluggable[pluggableId].pluggable_settings.power = power
        yangInstance.pluggables.pluggable[pluggableId].pluggable_settings.config_time = CONFIG_TIME_MAX
        print(f"Initialized {pluggableId} with frequency {frequency}, power {power}, and config_time {CONFIG_TIME_MAX}")

    print("Full YANG Model State:", yangInstance.get())
    return yangInstance

def findData(xpath, yangInstance, pluggableId):
    """
    @brief  Directly fetch settings from pluggableId
    @param  xpath The xpath to use for searching.
    @param  yangInstance The YANG instance.
    @param  pluggableId Transceiver ID.
    @return settings The settings of this transceiver.
    """
    if DEBUG:
        print(f"Processing XPath: {xpath}")
        print("Full YANG Model:", yangInstance.get())
 
    try:
        # Check if the raw dictionary contains the expected data
        rawModel = yangInstance.get()
        pluggable = rawModel['pluggables']['pluggable'].get(pluggableId, {})
        settings = pluggable.get('pluggable-settings', {})
        print("Raw settings accessed:", settings)
        return settings  # Return as a dictionary
    except Exception as ex:
        print(f"Error retrieving data: {ex}")
        return None

def generateGetReply(settings):
    """
    @brief  Generates an XML RPC reply for the pluggable settings.
    @param  The settings
    @return The XML RPC reply for the GET command.
    """
    if DEBUG:
        print("Settings passed to generateGetReply:", settings)
 
    try:
        # Check if settings is a dictionary
        if isinstance(settings, dict):
            frequency = settings.get('frequency', 'Unknown')
            power = settings.get('power', 'Unknown')
            configTime = settings.get('config_time', 'Unknown')
        else:
            # Assume settings is an object with attributes
            frequency = settings.frequency
            power = settings.power
            configTime = settings.config_time
 
        return f"""
            <rpc-reply xmlns="{NAMESPACE}" message-id="1">
                <data>
                    <pluggable-settings xmlns="urn:pluggable-config:1.0">
                        <frequency>{frequency}</frequency>
                        <power>{power}</power>
                        <config_time>{configTime}</config_time>
                    </pluggable-settings>
                </data>
            </rpc-reply>
        """
    except Exception as ex:
        print(f"Error generating RPC reply: {ex}")
        return f"""
            <rpc-reply xmlns="{NAMESPACE}" message-id="1">
                <rpc-error>
                    <error-type>application</error-type>
                    <error-tag>operation-failed</error-tag>
                    <error-message>Error generating response: {ex}</error-message>
                </rpc-error>
            </rpc-reply>
        """

def updateFrequency(pluggableId, frequency, yangInstance):
    """
    @brief  Validates and updates the frequency for the given pluggableId in the yangInstance.
    @param  pluggableId Transceiver ID.
    @param  frequency Transceiver frequency.
    @param  yangInstance The YANG instance.
    """
    if frequency is not None:
        frequency = int(frequency)
        print(f"Found frequency: {frequency}")
        if FREQ_LOW <= frequency <= FREQ_HIGH:  # Check the range constraint
            print(f"Will update frequency to: {frequency}")
        else:
            raise ValueError(f"Frequency value {frequency} is out of the allowed range ({FREQ_LOW}) to {FREQ_HIGH}).")

        # Set the actual laser frequency here on the pluggable (in GHz) using the command-line C++ wopen tool
        #threadId = executeCommandInThread(WOPEN_TOOL, f"--laser-freq {frequency}")
        threadId = executeCommandInThread(WOPEN_TOOL, f"{frequency} 1")
        waitForAllThreads()
        for tId, output in threadOutputs.items():
            if(tId == threadId):
                print(f"Thread ID: {tId}, Output: {output['output']}")
                updateConfigTime(pluggableId, output['output'], yangInstance)

        yangInstance.pluggables.pluggable[pluggableId].pluggable_settings.frequency = frequency
    else:
        print("Frequency not found or is None.")

def updatePower(pluggableId, power, yangInstance):
    """
    @brief  Validates and updates the power for the given pluggableId in the yangInstance.
    @param  pluggableId Transceiver ID.
    @param  power Transceiver port power.
    @param  yangInstance The YANG instance.
    """
    if power is not None:
        power = float(power)
        print(f"Found power: {power}")
        if POW_LOW <= power <= POW_HIGH:  # Check the range constraint
            print(f"Will update power to: {power}")
        else:
            raise ValueError(f"Power value {power} is out of the allowed range ({POW_LOW} to {POW_HIGH}).")
        
        # Set the actual laser power here on the pluggable (in dBm) using the command-line C++ wopen tool
        threadId = executeCommandInThread(WOPEN_TOOL, f"--tx-power {power}")
        waitForAllThreads()
        for tId, output in threadOutputs.items():
            if tId == threadId:
                print(f"Thread ID: {tId}, Output: {output['output']}")
                updateConfigTime(pluggableId, output['output'], yangInstance)

        yangInstance.pluggables.pluggable[pluggableId].pluggable_settings.power = power
    else:
        print("Power not found or is None.")

def updateConfigTime(pluggableId, configTime, yangInstance):
    """
    @brief  Validates and updates the power for the given pluggableId in the yangInstance.
    @param  pluggableId Transceiver ID.
    @param  configTime Configuration time.
    @param  yangInstance The YANG instance.
    """
    if configTime is not None:
        configTime = round(float(configTime), 2)
        
        if configTime <= CONFIG_TIME_MAX:  # Check the range constraint
            # Set the actual config_time here on pluggable's YANG state
            yangInstance.pluggables.pluggable[pluggableId].pluggable_settings.config_time = configTime
            print(f"Updated config_time to: {configTime}")
        else:
            print("config_time not found or is None.")

def generateEditConfigReply(request, yangInstance):
    """
    @brief  Handles edit-config RPC requests to set 'frequency' and 'power' values.
    @param  request XML request.
    @param  yangInstance The YANG instance.
    @return The XML RPC OK response including the feedback.
    """
    # Parse the XML request to extract the pluggableId
    root = ETree.fromstring(request)
    pluggableIdElement = root.find(".//{urn:pluggable-config:1.0}pluggable-id")
    pluggableId = pluggableIdElement.text

    try:
        # Parse the XML to extract frequency and power values
        root = ETree.fromstring(request)
        
        # Use namespace map to handle prefixes
        namespaces = {
            "nc": "urn:ietf:params:xml:ns:netconf:base:1.0",
            "cfg": "urn:pluggable-config:1.0",
        }
        
        config = root.find(".//nc:config", namespaces)
        if config is not None:
            print(f"Found <config>: {ETree.tostring(config, encoding ='unicode')}")

            # Find frequency and power using the correct namespace
            frequency = config.findtext(".//cfg:frequency", namespaces = namespaces)
            power = config.findtext(".//cfg:power", namespaces = namespaces)

            # Validate and update the frequency, power, and implicitly the config_time 
            updateFrequency(pluggableId, frequency, yangInstance)
            #updatePower(pluggableId, power, yangInstance)

            if updatePluggableHistory(pluggableId, yangInstance):
                print(f"Successfully updated settings for pluggableId: {pluggableId}")

            # Return success response including the feedback
            feedback = round(yangInstance.pluggables.pluggable[pluggableId].pluggable_settings.config_time, 2)
            return f"""
                <rpc-reply xmlns="{NAMESPACE}" message-id="1" feedback="{feedback}">
                    <ok/>
                </rpc-reply>
            """
        else:
            print("<config> element not found.")
            raise ValueError("No valid configuration found in the request.")
    except Exception as ex:
        print(f"Error handling edit-config request: {ex}")
        return f"""
            <rpc-reply xmlns="{NAMESPACE}" message-id="1">
                <rpc-error>
                    <error-type>application</error-type>
                    <error-tag>operation-failed</error-tag>
                    <error-message>Error applying configuration: {ex}</error-message>
                </rpc-error>
            </rpc-reply>
        """

def handleRpcRequest(request, yangInstance):
    """
    @brief  Handles RPC requests.
    @param  request XML RPC request.
    @param  yangInstance The YANG instance.
    @return The XML RPC GET or EDIT-CONFIG request.
    """
    try:
        # Parse the RPC XML
        root = ETree.fromstring(request)
        if root.find(f"{{{NAMESPACE}}}get") is not None:
            rpcType = RPC_TYPE.GET
        elif root.find(f"{{{NAMESPACE}}}edit-config") is not None:
            rpcType = RPC_TYPE.EDIT_CONFIG
        else:
            rpcType = RPC_TYPE.UNSUPPORTED

        if rpcType == RPC_TYPE.GET:
            print("Detected 'get' RPC request.")
            
            # Parse the XML request to extract the pluggableId
            root = ETree.fromstring(request)
            pluggableId = root.find(".//*[@select]").attrib['select'].split("[pluggable-id='")[1].split("']")[0]
            print(f"Extracted pluggable-id: {pluggableId}")
            
            try:
                # Fetch the settings using the extracted pluggableId
                settings = findData("", yangInstance, pluggableId)
            except Exception as e:
                print(f"Error processing the request: {e}")
                settings = None
            
            if settings:
                return generateGetReply(settings)
            else:
                print("No data found to respond with.")
                return f"""
                    <rpc-reply xmlns="{NAMESPACE}" message-id="1">
                        <rpc-error>
                            <error-type>application</error-type>
                            <error-tag>operation-not-supported</error-tag>
                            <error-message>No data found for the request.</error-message>
                        </rpc-error>
                    </rpc-reply>
                """
        elif rpcType == RPC_TYPE.EDIT_CONFIG:
            print("Detected 'edit-config' RPC request.")
            return generateEditConfigReply(request, yangInstance)
        else:
            print("Unsupported RPC operation.")
            return f"""
                <rpc-reply xmlns="{NAMESPACE}" message-id="1">
                    <rpc-error>
                        <error-type>application</error-type>
                        <error-tag>operation-not-supported</error-tag>
                        <error-message>Unsupported RPC operation.</error-message>
                    </rpc-error>
                </rpc-reply>
            """
    except Exception as ex:
        print(f"Error parsing RPC request: {ex}")
        return f"""
            <rpc-reply xmlns="{NAMESPACE}" message-id="1">
                <rpc-error>
                    <error-type>application</error-type>
                    <error-tag>operation-failed</error-tag>
                    <error-message>Invalid RPC request: {ex}</error-message>
                </rpc-error>
            </rpc-reply>
        """

def startNetconfServer(yangInstance):
    """
    @brief  Starts the Netconf server.
    @param  yangInstance The YANG instance.
    """
    if os.path.exists(socketPath):
        os.remove(socketPath)
    serverSocket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    serverSocket.bind(socketPath)
    serverSocket.listen(20)
    print(f"Netconf server running at {socketPath}. Waiting for connections...")
    try:
        while not exitApplication:
            clientSocket, _ = serverSocket.accept()
            with clientSocket:
                print("Connection accepted.")
                data = clientSocket.recv(BUFFER_SIZE)
                if data:
                    response = handleRpcRequest(data.decode(), yangInstance)
                    print(f"Sending RPC response:\n{response}")
                    clientSocket.sendall(response.encode())
    except KeyboardInterrupt:
        print("Shutting down server...")
    finally:
        serverSocket.close()
        if os.path.exists(socketPath):
            os.remove(socketPath)

def main():
    global exitApplication
    initSocketPath(SOCKET_FOLDER)
    signal.signal(signal.SIGINT, signalHandler)
    yangInstance = initializeYangModel()
    startNetconfServer(yangInstance)

if __name__ == "__main__":
    main()
