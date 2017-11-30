# Design document of Total Reduce

## Goal of Total Reduce
The goal of Total Reduce aims to solve the following problems:
1. There are many different gradients needs allreduce, each start at different time and is needed at different time, too.
2. On different node, gradient might be ready in different order, making coordination needed.
3. Preemptive communication is needed for certain message.
4. Computation might get out of sync, making communication skewed.

Total Reduce look all gradients allreduce as a whole data needs to be sync at different time, and try to provide an optimal solution that is robost and resillent.

## Idea of Total Reduce
### Use Ring as basic infrastructure
Using Ring, each node have predictable communication target and source, this makes some of the ideas like self guided message below easy to implement.
### Break down of allreduce algorithm
Rather than look allreduce as a black box and call communication library for each different message.  Total Reduce breakdown allreduce algorithm into element operation (isend/irecv).  By breakdown to more element operation, we get better granularity for different purpose:
* communication preempt.  One allreduce operation can be preempted for other allreduce operation after each element operation.
* asynchronous progress.  (Note this is not asynchronous SGD).  Asynchronous progress means each node does not need to wait for other nodes to reach the same stage to be preempted, which reduces synchronization cost and also improves network utilization.
### self guided message
Each isend belongs to certain step of certain allreduce operation.  Before isend, a header message describing the step/operation id will be sent over first.  So receiever would figure out what the message is for by check the header, and guide the irecv operation to proper buffer.  So each node does not have to at the same page when doing isend/irecv operation.  And there is no need to sync.
### collective operation id
Each collective operation will have its unique id, this id will be used to identify the self guided message.   This id is also used to manage ahead-of-time message, where the message is ready on one node but another node had not been ready yet.
### reduce thread
There will be two threads, one threads dadecated for communication, doing the irecv operation to fill buffer.  Another thread will dadecate for computation, doing the reduce operation whenever two buffer is ready.

## Data structure
### message list
Message list store the message awaiting to be delievered, and which step it is on each node.  If we have a two layer DNN, then we have two message on each node.  The message list may contain two messages.

It it not necessary to maintain a global status of each message.  Each message have its own state on each node.  The state (step) might be different on different nodes.  So there is no need to sync state between nodes, thus lower sync cost.

Each message in the list has an ID associate to it.  For the same message across nodes, they have same message ID.

Fields of message:
* ID        -- ID of message
* Op        -- Operation, currently only allreduce is used
* Size      -- Size of message
* InBuf     -- InBuf of message
* OutBuf    -- OutBuf of message
* DataType  -- DataType of message
* Algo      -- The algorithm usage for this message, currently only ring is used
* State     -- State of message, this relates to which algorithm is used

### isend message head
isend message head is a small message that describe the next message that actually contains the data, it contains the following information:
* ID        -- Which ID this isend message belongs to
* Size      -- the size of this isend message
* State     -- the state of this isend message

One may wonder why the size of this isend message needs to be sent over, the target should know the size, because target knows the state and can see message list.  But consider one node is ready to do allreduce and already start sending message.  The target n ot even started allreduce, so when it gets an ID, it does not know what to do with it.   The size allows next node allocate a temporary memory and store the message, then copy the message over when necessary.  For same reason, this node may send multiple message related to same ID, so state can be used to distinguish them.  Algo and OP does not needs to be send over because next node wont do anything about the message before it get Op and Algo from its own.

### floating message list
floating message are message that does not have a message id associate with it.  They need to be resolve until message with message id start to do allreduce
each floating message contains the following information, they are basiclly the info from isend message head
* ID        -- Which ID this message belongs to
* Size      -- the size of this isend message
* State     -- the state of this message
* buf       -- the buffer address containing this message

### computation list
message awating to be added together
* ID        -- Which ID this message belongs to
* Size      -- The size of this message
* DataType  -- the data type, so computation module know how to do the sum operation
* ReadBuf0  -- the first buffer
* ReadBuf1  -- the second buffer
* WriteBuf  -- the write back buffer, note the buffer does not need to be different, it could be ReadBuf0 or ReadBuf1

## Threads
There are two threads running on each node, one for communication, another for computation.  Combine these two threads together might work, but needs careful design/implementation.  Currently, two threads is a simpler design

## Algorithm
### When new gradient is ready
* For gradient of each different layer, assign a unique id to this graident
When a new gradient is ready, put it to payload list

### When gradient is needed
Check whether its payload is done, if true, return ready, else wait if necessary, or return not ready.

### the big loop
void big_loop(void)
{
    mark no message is sending
    mark no message is recving
    for (;;) {
        if no message is sending {
            // progress send part
            struct msg *payload = payload_pick_ready();
            if (payload!= NULL) {
                mark payload not ready for send
                payload_send_step();
            }
        }
        if no message is receiving {
            start head recv
        }

        go over all floating message, reduce if they no longer needs to be float

        if message is receiving {
            probe_recv_status
            if head recived {
                decide which payload is about to be received
                if payload found according to id in message head
                    start body recv for payload buffer
                else // floating message
                    allocate a piece of memory for the message buffer
                    start body recv to allocated message buffer
            }
            if body recived {
                if (message is not floating) {
                    do reduce/(overwrite) action // overwrite may be null action
                    mark message ready for send again
                    mark no message is receiving
                    mark payload as done if necessary
                } else {
                    put message in float list, set its stage
                }
            }
        }
        if message is sending {
            probe_send_status to make progress
            if message sent
                mark no message is sending
        }
    }
    sleep a while
}

