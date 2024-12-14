'''plotjoints.py

   Plot the /joint_states recorded in the ROS2 bag.
'''

import rclpy
import numpy as np
import matplotlib.pyplot as plt

import glob, os, sys

from rosbag2_py                 import SequentialReader
from rosbag2_py._storage        import StorageOptions, ConverterOptions
from rclpy.serialization        import deserialize_message

from sensor_msgs.msg            import JointState


#
#  Plot the Joint Data
#
def plotjoints(jointmsgs, t0, bagname, jointname='all'):
    # Process the joint messages.
    names = jointmsgs[0].name

    sec  = np.array([msg.header.stamp.sec     for msg in jointmsgs])
    nano = np.array([msg.header.stamp.nanosec for msg in jointmsgs])
    t = sec + nano*1e-9 - t0

    pos = np.array([msg.position for msg in jointmsgs])
    vel = np.array([msg.velocity for msg in jointmsgs])

    # Extract the specified joint.
    if jointname != 'all':
        # Grab the joint index.
        try:
            index = int(jointname)
        except Exception:
            index = None
        try:
            if index:
                jointname = names[index]
            else:
                index = names.index(jointname)
        except Exception:
            raise ValueError("No data for joint '%s'" % jointname)

        # Limit the data.
        names = [names[index]]
        pos   = pos[:,index]
        vel   = vel[:,index]

    # Re-zero time.
    tstart = min(t)
    print("Starting at time ", tstart)
    t = t - tstart


    # Create a figure to plot pos and vel vs. t
    fig, axs = plt.subplots(2, 1)

    # Plot the data in the subplots.
    axs[0].plot(t, pos)
    axs[0].set(ylabel='Position (rad)')
    axs[1].plot(t, vel)
    axs[1].set(ylabel='Velocity (rad/sec)')

    # Connect the time.
    axs[1].set(xlabel='Time (sec)')
    axs[1].sharex(axs[0])

    # Add the title and legend.
    axs[0].set(title="Joint Data in '%s'" % bagname)
    axs[0].legend(names)

    # Draw grid lines and allow only "outside" ticks/labels in each subplot.
    for ax in axs.flat:
        ax.grid()
        ax.label_outer()


#
#  Main Code
#
def main():
    # Grab the arguments.
    jointname = 'all'    if len(sys.argv) < 3 else sys.argv[2]
    bagname   = 'latest' if len(sys.argv) < 2 else sys.argv[1]

    # Check for the latest ROS bag:
    if bagname == 'latest':
        # Report.
        print("Looking for latest ROS bag...")

        # Look at all bags, making sure we have at least one!
        dbfiles = glob.glob('*/*.db3')
        if not dbfiles:
            raise FileNoFoundError('Unable to find a ROS2 bag')

        # Grab the modification times and the index of the newest.
        dbtimes = [os.path.getmtime(dbfile) for dbfile in dbfiles]
        i = dbtimes.index(max(dbtimes))

        # Select the newest.
        bagname = os.path.dirname(dbfiles[i])

    # Report.
    print("Reading ROS bag '%s'"  % bagname)
    print("Processing joint '%s'" % jointname)


    # Set up the BAG reader.
    reader = SequentialReader()
    try:
        reader.open(StorageOptions(uri=bagname, storage_id='sqlite3'),
                    ConverterOptions('', ''))
    except Exception as e:
        print("Unable to read the ROS bag '%s'!" % bagname)
        print("Does it exist and WAS THE RECORDING Ctrl-c KILLED?")
        raise OSError("Error reading bag - did recording end?") from None

    # Get the starting time.
    t0 = reader.get_metadata().starting_time.nanoseconds * 1e-9 - 0.01

    # Get the topics and types:
    print("The bag contain message for:")
    for x in reader.get_all_topics_and_types():
        print("  topic %-20s of type %s" % (x.name, x.type))


    # Pull out the relevant messages.
    jointmsgs = []
    while reader.has_next():
        # Grab a message.
        (topic, rawdata, timestamp) = reader.read_next()

        # Pull out the deserialized message.
        if   topic == '/joint_states':
            jointmsgs.append(deserialize_message(rawdata, JointState))


    # Process the joints
    if jointmsgs:
        print("Plotting joint data...")
        plotjoints(jointmsgs, t0, bagname, jointname)
    else:
        raise ValueError("No joint data!")

    # Show
    plt.show()


#
#   Run the main code.
#
if __name__ == "__main__":
    main()
