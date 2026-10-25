# A harder held-out opponent

Declared before examining this baseline's results. Largest-piece play resolves
all ties at random; policies might exploit that weakness. Probe a second,
handcrafted baseline that first maximizes piece area and then maximizes
`own empty corner anchors - opponent empty corner anchors` in the resulting
position. Resolve remaining ties uniformly. An anchor excludes occupied cells
and cells adjacent to one's own pieces along an edge. It need not fit every
remaining shape, so the count is only a geometric mobility proxy.

This baseline uses one successor-state calculation per candidate and no learned
weights. It is more expensive than random tie-breaking. Do not call it an
optimal opponent or a trained agent, and do not assume it is stronger until
measured. It is held out from the existing training curricula and checkpoint
selection.

First benchmark: 100 games per seat with seed 790000, frontier against largest,
and selected largest-margin-142 against frontier. Then test the selected longer
run against frontier after its checkpoint is chosen by its existing validation
criterion. Save JSON results separately. A weakness here would motivate a new
training experiment, not invalidate the earlier named-baseline results.
