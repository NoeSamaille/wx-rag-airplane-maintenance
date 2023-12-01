# RAG example for airplane maintenance with watsonx.ai

Simple implementation of RAG using watsonx.ai, capturing the chat history to keep track of the conversation context and answer follow up questions.

## Prereqs

Python `3.10` or `3.11`.

## Quick Start

```sh
python -m pip install -r requirements.txt
wget https://sportflyingusa.com/wp-content/uploads/2020/08/Maintenance-Manual.pdf
export WX_API_KEY=<ibm-cloud-api-key>
export WX_PROJECT_ID=<watsonx-ai-project-id>
export WX_URL=https://us-south.ml.cloud.ibm.com # https://eu-de.ml.cloud.ibm.com for Frankfurt
python main.py
```

## Expected output

```
❯ python main.py
Loading models...
Loading documents...
Splitting documents...
Vectorizing documents...
> What tools do I need to remove aileron control bellcrank in the wing?
7/16 inch wrench
Screwdriver
Cutting pliers
> How do I disassemble it?
6.3.10  Removal of aileron control bellcrank in the wing  
Type of maintenance: heavy  
Authorization to perform:  
- Repairman (LS -M) or Mechanic (A&P) . 
Tools needed:  
- wrench size 7/16 in  
- screwdriver  
 
The bellcrank  is located on the bracket in the position of the  main sper next to the  rear rib No. 
1.  
Disassembly is identical for the left and the right wing  (see Fig. 6 -11). 
(a) Remove the cover (1) from access hole on the lower side of the wing .  
(b) Remove the rods (3) and (4) from the bellcrank arm - unscrew the nuts and remove the 
bolts (5) and (6).  
(c) Remove the bellcrank (7) from the wing - unscrew the nut and remove the bolt (8).

Step (a) Remove the cover (1) from access hole on the lower side of the wing .
Step (b) Remove the rods (3) and (4) from the bellcrank arm - unscrew the nuts and remove the bolts (5) and (6).
Step (c) Remove the bellcrank (7) from the wing - unscrew the nut and remove the bolt (8).
> How do I remove the canopy?
3.3.1 Canopy removal
Type of maintenance: line
Authorization to perform: Sport pilot or higher
Tools needed:
- Socket wrench 7/16”
- Screwdriver
- Pliers

Follow the Fig. 3 -3 at removing of the canopy:
(a) Open the canopy (1).
(b) Remove securing springs from the gas strut rod ends (2).
(c) Disconnect gas struts (3) on both sides of the canopy (1).
(d) Disconnect hinge bolt nuts (4).
(e) Remove the hinge bolts (5).
(f) Remove the canopy (1) and store it in a safe place so that windscreen damage cannot occur.

Therefore, to remove the canopy of an aircraft, you should:

1. Open the canopy.
2. Remove the securing springs from the gas strut rod ends.
3. Disconnect the gas struts on both sides of the canopy.
4. Disconnect the hinge bolt nuts.
5. Remove the hinge bolts.
6. Remove the canopy and store it in a safe place.
> What tools do I need for that?
 Based on the given context, the tools required to remove an aircraft's canopy are:

* Socket wrench size 7/16
* Screwdriver
* Pliers

These tools are mentioned in the section 3.3.1 Canopy removal, which describes the steps to remove the canopy.
```
