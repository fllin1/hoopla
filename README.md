# Hoopla

Modern search engine techniques like keyword, vector, semantic and LLM-enhanced search. In this project, we implement different search techniques, from a simple keyword search up to a fully functional Retrieval Augmented Generation (RAG) pipeline using Gemini API.


## Installation

```bash
git clone https://github.com/fllin1/hoopla
cd hoopla/

uv sync
uv pip install -e .

source .venv/bin/activate # On a Linux distro
```

## Run

### Keywords search

For our tests, we used data on movies. You can get if from the following [link](https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/course-rag-movies.json) (credits to [Boot.dev](https://boot.dev)) that you should save in the file-path `./data/movies.json`. 

You will also need a list of stop-words, we used the following [stop-words](#stop-words) and save them in the file-path `./data/stopwords.txt`

### Semantic search

## Appendix

### Stop-words

```txt
a
about
above
after
again
against
ain
all
am
an
and
any
are
aren
aren't
as
at
be
because
been
before
being
below
between
both
but
by
can
couldn
couldn't
d
did
didn
didn't
do
does
doesn
doesn't
doing
don
don't
down
during
each
few
for
from
further
had
hadn
hadn't
has
hasn
hasn't
have
haven
haven't
having
he
he'd
he'll
he's
her
here
hers
herself
him
himself
his
how
i
i'd
i'll
i'm
i've
if
in
into
is
isn
isn't
it
it'd
it'll
it's
its
itself
just
ll
m
ma
me
mightn
mightn't
more
most
mustn
mustn't
my
myself
needn
needn't
no
nor
not
now
o
of
off
on
once
only
or
other
our
ours
ourselves
out
over
own
re
s
same
shan
shan't
she
she'd
she'll
she's
should
should've
shouldn
shouldn't
so
some
such
t
than
that
that'll
the
their
theirs
them
themselves
then
there
these
they
they'd
they'll
they're
they've
this
those
through
to
too
under
until
up
ve
very
was
wasn
wasn't
we
we'd
we'll
we're
we've
were
weren
weren't
what
when
where
which
while
who
whom
why
will
with
won
won't
wouldn
wouldn't
y
you
you'd
you'll
you're
you've
your
yours
yourself
yourselves
```
