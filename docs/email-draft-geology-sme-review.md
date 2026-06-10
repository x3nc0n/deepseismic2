# Email Draft — Geology SME Review Request

**To:** John Spaid (spaidjsr78@gmail.com)  
**Subject:** Picking your brain — modernizing seismic interpretation with AI

---

Hey John,

I'm working on a proof-of-concept that aims to make seismic interpretation more accessible and affordable — essentially taking the workflows that traditionally require expensive software (think Petrel, Kingdom) and specialized storage hardware, and rebuilding them using modern cloud tools and AI assistants.

The idea isn't to replace the geologist — it's to give less-experienced interpreters (or geologists working outside their home basin) the kind of contextual support that currently lives in the heads of people like you.

I'm using Equinor's Volve dataset (North Sea, Viking Graben, Hugin Formation) as the test case. I'd love your perspective on a few things:

**1. The interpretation workflow itself**

When you sit down to interpret a new 3D seismic survey in an unfamiliar basin, what's your mental checklist? I'm trying to map out the "series of specific tasks" that an experienced interpreter follows — from initial data QC through to a final interpretation product. Where are the decision points that require judgment vs. the steps that are largely mechanical?

**2. Regional knowledge as a barrier**

How much of what makes a good interpretation "good" is regional knowledge (depositional models, expected fault styles, stratigraphic framework) vs. transferable geophysical/geological skill? If you were handed a dataset from offshore West Africa instead of the North Sea, what would you need to get up to speed?

**3. Where AI assistance would actually help**

Imagine you had an assistant that could:
- Automatically suggest which horizons to pick based on the regional stratigraphy
- Flag areas where the seismic facies don't match what's geologically expected
- Generate a first-pass facies classification that you then validate and edit
- Provide context about the depositional environment ("this looks like a prograding shoreface because...")
- Summarize your interpretation progress and generate handoff notes

Which of those would genuinely save time or improve quality? Which would you not trust? Are there other things you wish an assistant could do?

**4. Facies classification priorities**

For a North Sea Jurassic shallow marine setting (like Volve/Hugin Formation), what facies would you expect to see on seismic, and which ones actually matter for a development decision? I'm trying to scope what the ML model should attempt to classify vs. what's below seismic resolution or not practically useful.

**5. Validation — how do you know it's right?**

When you review someone else's interpretation, what do you look for? What are the telltale signs of a bad interpretation (geologically impossible juxtapositions, missed faults, etc.)? This helps us design the AI's self-check logic.

No rush — just thinking out loud at this stage. Happy to jump on a call if that's easier than typing it all out.

Thanks,
Jos
