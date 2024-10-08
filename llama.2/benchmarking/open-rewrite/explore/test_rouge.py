from ignite.metrics import RougeL

m = RougeL(multiref="best")

candidate = "Section 612 of the Fair Credit Reporting Act (15 U.S.C. 1681j) is amended as follows:\n(1) In subsection (f)(1), in the matter before subparagraph (A), by striking \"through (d)\" and inserting \"through (h)\".\n(2) At the end of the subsection, by adding the following:\n\"(h) Free Disclosures Connecting With Credit Freeze.--In addition to the free yearly disclosure required under subsection (a)(1)(A), each buyer reporting agency that maintains a file on a buyer who requests a credit freeze under section 605A(i) can make all disclosures in accordance with section 609 once during any year without charge to the buyer if the buyer makes a request under section 609.".split()
references = "Section 612 of the Fair Credit Reporting Act (15 U.S.C. 1681j) is amended--(1) in subsection (f)(1), in the matter preceding subparagraph (A), by inserting ``or subsection (h)'' after ``through (d)''; and(2) by adding at the end the following:``(h) Free Disclosures in Connection With Credit Freeze.--In addition to the free annual disclosure required under subsection (a)(1)(A), each consumer reporting agency that maintains a file on a consumer who requests a credit freeze under section 605A(i) shall make all disclosures pursuant to section 609 once during any 12-month period without charge to the consumer if the consumer makes a request under section 609.''.".split(),


m.update(([candidate], [references]))

print(m.compute())