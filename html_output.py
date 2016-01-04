""" copy of http://www.scipy-lectures.org/advanced/scikit-learn/
Author : Bernd
Date : Dec. 23, 2015
"""
import sys

class html_picture_summary():
    def __init__(self, of_name):
        self._output_file=of_name
        self._header = ""
        self._body = ""

    def header(self,title, CSS):
        """
        produces the header of an html file. CSS is a list of style sheets
        """
        l = []
        l[len(l):] = ["<head><title>%s</title>" % title]
        l[len(l):] = ['<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">']
        for c in CSS:
            l[len(l):] = ['<link rel="stylesheet" type="text/css" href="%s">' % c]
        l[len(l):] = ["</head>"]
        self._header = "\n".join(l)

    def body(self, title):
        l = []
        l[len(l):] = ["<body>"]
        l[len(l):] = ["<h1>%s</h1>" % title]
        l[len(l):] = ["</body>"]
        self._body = "\n".join(l)

    def all_html(self):
        """
        produces the actual html code of the whole document. Requires self._header and self._body to be set.
        """
        l = []
        l[len(l):] = ["<!DOCTYPE HTML PUBLIC '-//W3C//DTD HTML 4.01 Transitional//EN'>"]
        l[len(l):] = ["<html>"]
        l[len(l):] = [self._header]
        l[len(l):] = [self._body]
        l[len(l):] = ["</html>"]
        return "\n".join(l)


    def loop(self):
        f = open(self._output_file, 'w')
        title = "Yippie!"
        h.header(title, ["Nachmieter.css"])
        h.body(title)
        f.write(h.all_html())

        f.close()
        if f.closed: print "wrote to: %s" % self._output_file
        else:
            print "something went wrong with %s" % self._output_file
            sys.exit(-1)

def main():
    output_file = "mytest.html"
    h = html_picture_summary(output_file)
    h.loop()
    

if __name__ == "__main__":
    status = main()
    sys.exit(status)
