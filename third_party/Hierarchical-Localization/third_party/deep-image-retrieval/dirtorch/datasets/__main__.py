import os
import sys
import pdb 
from nltools.gutils.pyplot import *


def viz_dataset(db, nr=6, nc=6):
    ''' a convenient way to vizualize the content of a dataset.
        If there are queries, it will show the ground-truth for each query.
    '''
    pyplot(globals())
    
    try:
        query_db = db.get_query_db()
        
        LABEL = ['null', 'pos', 'neg']
        
        for query in range(query_db.nimg):
            figure("Query")
            subplot_grid(20, 1)
            pl.imshow(query_db.get_image(query))
            pl.xlabel('Query #%d' % query)
            
            pl_noticks()
            gt = db.get_query_groundtruth(query)
            ranked = (-gt).argsort()
            
            for i,idx in enumerate(ranked):
                if i+2 > 20: break
                subplot_grid(20, i+2)
                pl.imshow(db.get_image(idx))
                label = gt[idx]
                pl.xlabel('#%d %s' % (idx, LABEL[label]))
                pl_noticks()
            pdb.set_trace()
    
    except NotImplementedError:
        import numpy as np
        pl.ion()
        
        R = 1
        nr = nr // R
        
        def show_img(r, c, i):
            i = np.random.randint(len(db))
            
            pl.subplot(R*nr,nc,(R*r+0)*nc+c+1)
            img = db.get_image(i)
            print(i, db.get_key(i), "%d x %d" % img.size)
            pl.imshow(img)
            pl.xticks(())
            pl.yticks(())
            if db.has_label():
                pl.xlabel(db.get_label(i))

        pl.figure()
        pl.subplots_adjust(0,0,1,1,0.02,)
        n = 0
        while True:
            pl.clf()
            for r in range(nr):
              for c in range(nc):
                show_img(r,c,n)
                n += 1
            pdb.set_trace()



if __name__ == '__main__':
    from .__init__ import create

    args = sys.argv[1:]
    if not args:
        print("Error: Provide a db_cmd to this script"); 
        exit()
    
    db = args.pop(0)
    print("Instanciating dataset '%s'..." % db)
    
    db = create(db)
    print(db)
        
    viz_dataset(db)
