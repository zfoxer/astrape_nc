/**
 *  Astrape for Netconf (c) 2024-2025 by Constantine Kyriakopoulos, zfox@users.sourceforge.net.
 *  Compile with 'clang++ -std=c++23 -o wopen_sim wopen_sim.cpp'
 *
 *  A simple simulator for a tool that configures transceiver laser frequencies. Gets as input a
 *  frequency slot and writes to the standard output the corresponding configuration time.
 */

/**
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <random>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <unordered_map>
#include <cmath>
#include <fstream>


/**
 *  Generates configuration times based on a log-normal distribution.
 *
 *  @param frequencySlot  The frequency to use for setting the transceiver laser at.
 *  @param numOfValues  Number of configuration times to produce.
 *  @param slotDistributions  Contains for every slot, the mean and standard deviation.
 */
void generateTimes(int frequencySlot, int numOfValues,
        const std::unordered_map<int, std::pair<double, double>>& slotDistributions)
{
    // Check if the frequency slot exists in the distribution map
    if (slotDistributions.find(frequencySlot) == slotDistributions.end())
    {
        std::cerr << "Error: Frequency slot " << frequencySlot << " not found in distribution data.\n";
        return;
    }

    // Retrieve the mean and standard deviation for the given frequency slot
    double timeMean = slotDistributions.at(frequencySlot).first;
    double timeStdDev = slotDistributions.at(frequencySlot).second;
    // Convert mean and standard deviation to log-normal parameters
    double mu = std::log(timeMean) - 0.5 * std::pow(timeStdDev / timeMean, 2);
    double sigma = std::sqrt(std::log(1 + std::pow(timeStdDev / timeMean, 2)));

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::lognormal_distribution<> timeDist(mu, sigma);

    // Generate and print 'numOfValues' configuration times
    for(int i = 0; i < numOfValues; ++i)
        std::cout << timeDist(gen) << "\n";
}

/**
 *  Generation starts here.
 */
int main(int argc, char** argv)
{
    if(argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <frequencySlot> <number_of_values>\n";
        return EXIT_FAILURE;
    }

    int frequencySlot = std::atoi(argv[1]);
    int numOfValues = std::atoi(argv[2]);

    //  Define the mean and standard deviation for each frequency slot.
    //  Produced from the dataset.
    std::unordered_map<int, std::pair<double, double>> slotDistributions =
    {
        {191300, {3.5673262242918398, 0.10197645357176223}},
        {191400, {3.582677412882187, 0.07290916648048067}},
        {191500, {3.4786869181983473, 0.1011383244896039}},
        {191600, {3.983720769681577, 0.5763068100659825}},
        {191700, {4.534629206623522, 0.10845202692648127}},
        {191800, {3.3612826822872073, 0.22707099443225762}},
        {191900, {4.540865770999331, 0.09991432469653252}},
        {192000, {4.0489479338580345, 0.553058627624838}},
        {192100, {3.9723889915620565, 0.5493783112221067}},
        {192200, {4.003676955007723, 0.5938438523844344}},
        {192300, {4.5456963236028365, 0.08971355419953174}},
        {192400, {4.57768638569735, 0.10520795619891637}},
        {192500, {4.488659390254012, 0.10283021976387242}},
        {192600, {4.537509803848422, 0.0818422611265871}},
        {192700, {4.305613065471783, 0.22580320342074317}},
        {192800, {4.125345304817222, 0.58546454880053}},
        {192900, {4.134380691026164, 0.6450861619368289}},
        {193000, {4.6145538199678215, 0.12301048179097206}},
        {193100, {4.727596782215347, 0.166229865534735}},
        {193200, {4.5481082985538865, 0.14390186563265814}},
        {193300, {4.548160877152112, 0.09986612903129362}},
        {193400, {4.582510434240715, 0.10173507635703509}},
        {193500, {4.589989283964432, 0.08039081515705251}},
        {193600, {4.625584653434835, 0.059825717781708795}},
        {193700, {4.558311083972083, 0.09700297982067743}},
        {193800, {4.0908964661713405, 0.39207497278039327}},
        {193900, {4.528228502060346, 0.060776568646497164}},
        {194000, {4.013609312355294, 0.4537997999853553}},
        {194100, {4.430404794721973, 0.09652761497302578}},
        {194200, {4.507629361830334, 0.11879015792186062}},
        {194300, {4.505616903827665, 0.055383610344386915}},
        {194400, {3.993342992008915, 0.5066357423240982}},
        {194500, {4.590765362677227, 0.08297130193743742}},
        {194600, {4.4886051174433, 0.08000292280679686}},
        {194700, {4.541742602134247, 0.1430866059671114}},
        {194800, {4.514768422772212, 0.07768467455107683}},
        {194900, {4.481626015848277, 0.08189271709171975}},
        {195000, {4.427328259802256, 0.07043712466239796}},
        {195100, {4.517807210605417, 0.08988841367660816}},
        {195200, {4.516808604696495, 0.11241722090480011}},
        {195300, {4.515690726774643, 0.15169226228402363}},
        {195400, {4.571597355358047, 0.1833743682044003}},
        {195500, {4.0470779540430755, 0.5891767894736875}},
        {195600, {3.983728545097063, 0.5185427186735012}},
        {195700, {4.86810387669144, 0.15287041828363068}},
        {195800, {4.415630423073828, 0.10009632829935784}},
        {195900, {4.552815392243745, 0.08995334380753991}},
        {196000, {4.589763192489785, 0.19826935647619867}},
        {196100, {4.60749047429043, 0.1089497747442195}}
    };

    generateTimes(frequencySlot, numOfValues, slotDistributions);

    return EXIT_SUCCESS;
}
