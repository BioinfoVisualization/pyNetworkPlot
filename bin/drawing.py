""" Specific functions for figure drawing using pyCairo.

    These functions draw simple figures over a pyCairo context.
        * draw_circle
        * draw_square
        * draw_triangle_up
        * draw_triangle_down
        * draw_diamond

    Authors: Juan Sebastian Diaz Boada
             juan.sebastian.diaz.boada@ki.se

    09/07/22
"""
from math import pi
import cairo as cr
#---------------------------------------------------------------------------------------------------#
def draw_circle(ctx, coord,size,in_color = (1,1,1), line_color=(0,0,0),line_width = None):
    """ Draws a circle given the coordinates and size of its inscribing square.

    Parameters
        ----------
        ctx : Cairo Context
            Context where the circle is going to be drawn.
        coord : tuple
            Tuple with x and y coordinate of the top-left corner of the inscribing square
            which will hold the circle.
        size : float
            Length of the side of the inscribing square.
        in_color : tuple, optional
            3- tuple containing the filling color for the circle in RGB format.
            Default is white (1.0,1.0,1.0)
        line_color : tuple, optional
            3- tuple containing the color for the circle perimeter in RGB format.
            Default is black (0.0,0.0,0.0)
        line_width : float or None
            Thickness of the shape contour. If None, assigned to be 0.1*size.
    """
    if line_width == None:
        line_width = 0.1*size
    ctx.move_to(coord[0]+size,coord[1]+0.5*size)
    ctx.arc(coord[0]+0.5*size,coord[1]+0.5*size, 0.5*size, 0, 2*pi)
    ctx.close_path()
    ctx.set_source_rgb(in_color[0],in_color[1],in_color[2])
    ctx.fill_preserve()
    ctx.set_source_rgb(line_color[0],line_color[1],line_color[2])
    ctx.set_line_width(line_width)
    ctx.stroke()
#---------------------------------------------------------------------------------------------------#
def draw_square(ctx,coord,size,in_color = (1,1,1), line_color=(0,0,0),line_width = None):
    """ Draws a sqare given the top-left coordinate and its size.

    Parameters
        ----------
        ctx : Cairo Context
            Context where the square is going to be drawn.
        coord : tuple
            Tuple with x and y coordinate of the top-left corner of the square.
        size : float
            Length of the side of the square.
        in_color : tuple, optional
            3- tuple containing the filling color for the square in RGB format.
            Default is white (1.0,1.0,1.0)
        line_color : tuple, optional
            3- tuple containing the color for the square perimeter in RGB format.
            Default is black (0.0,0.0,0.0)
        line_width : float or None
            Thickness of the shape contour. If None, assigned to be 0.1*size.
    """
    if line_width == None:
        line_width = 0.1*size
    ctx.rectangle(coord[0],coord[1],size,size)
    ctx.set_source_rgb(in_color[0],in_color[1],in_color[2])
    ctx.fill_preserve()
    ctx.set_source_rgb(line_color[0],line_color[1],line_color[2])
    ctx.set_line_width(line_width)
    ctx.stroke()
#---------------------------------------------------------------------------------------------------#
def draw_triangle_down(ctx,coord,size,in_color = (1,1,1), line_color=(0,0,0),line_width = None):
    """ Draws a down-faced triangle given the coordinates and size of its inscribing square.

    Parameters
        ----------
        ctx : Cairo Context
            Context where the triangle is going to be drawn.
        coord : tuple
            Tuple with x and y coordinate of the top-left corner of the inscribing square
            which will hold the triangle.
        size : float
            Length of the side of the inscribing square.
        in_color : tuple, optional
            3- tuple containing the filling color for the triangle in RGB format.
            Default is white (1.0,1.0,1.0)
        line_color : tuple, optional
            3- tuple containing the color for the triangle perimeter in RGB format.
            Default is black (0.0,0.0,0.0)
        line_width : float or None
            Thickness of the shape contour. If None, assigned to be 0.1*size.
    """
    if line_width == None:
        line_width = 0.1*size
    ctx.move_to(coord[0],coord[1])
    ctx.line_to(coord[0]+size, coord[1])
    ctx.line_to(coord[0]+0.5*size, coord[1]+size)
    ctx.close_path()
    ctx.set_source_rgb(in_color[0],in_color[1],in_color[2])
    ctx.fill_preserve()
    ctx.set_source_rgb(line_color[0],line_color[1],line_color[2])
    ctx.set_line_width(line_width)
    ctx.stroke()
#---------------------------------------------------------------------------------------------------#
def draw_triangle_up(ctx,coord,size,in_color = (1,1,1), line_color=(0,0,0),line_width = None):
    """ Draws an u-faced triangle given the coordinates and size of its inscribing square.

    Parameters
        ----------
        ctx : Cairo Context
            Context where the triangle is going to be drawn.
        coord : tuple
            Tuple with x and y coordinate of the top-left corner of the inscribing square
            which will hold the triangle.
        size : float
            Length of the side of the inscribing square.
        in_color : tuple, optional
            3- tuple containing the filling color for the triangle in RGB format.
            Default is white (1.0,1.0,1.0)
        line_color : tuple, optional
            3- tuple containing the color for the triangle perimeter in RGB format.
            Default is black (0.0,0.0,0.0)
        line_width : float or None
            Thickness of the shape contour. If None, assigned to be 0.1*size.
    """
    if line_width == None:
        line_width = 0.1*size
    ctx.move_to(coord[0]+0.5*size,coord[1])
    ctx.line_to(coord[0]+size, coord[1]+size)
    ctx.line_to(coord[0], coord[1]+size)
    ctx.close_path()
    ctx.set_source_rgb(in_color[0],in_color[1],in_color[2])
    ctx.fill_preserve()
    ctx.set_source_rgb(line_color[0],line_color[1],line_color[2])
    ctx.set_line_width(line_width)
    ctx.stroke()
#---------------------------------------------------------------------------------------------------#
def draw_diamond(ctx,coord,size,in_color = (1,1,1), line_color=(0,0,0),line_width = None):
    """ Draws a diamond given the coordinates and size of its inscribing square.

    Parameters
        ----------
        ctx : Cairo Context
            Context where the diamond is going to be drawn.
        coord : tuple
            Tuple with x and y coordinate of the top-left corner of the inscribing square
            which will hold the diamond.
        size : float
            Length of the side of the inscribing square.
        in_color : tuple, optional
            3- tuple containing the filling color for the diamond in RGB format.
            Default is white (1.0,1.0,1.0)
        line_color : tuple, optional
            3- tuple containing the color for the diamond perimeter in RGB format.
            Default is black (0.0,0.0,0.0)
        line_width : float or None
            Thickness of the shape contour. If None, assigned to be 0.1*size.
    """
    if line_width == None:
        line_width = 0.1*size
    ctx.move_to(coord[0]+0.5*size,coord[1])
    ctx.line_to(coord[0]+size, coord[1]+0.5*size)
    ctx.line_to(coord[0]+0.5*size, coord[1]+size)
    ctx.line_to(coord[0], coord[1]+0.5*size)
    ctx.close_path()
    ctx.set_source_rgb(in_color[0],in_color[1],in_color[2])
    ctx.fill_preserve()
    ctx.set_source_rgb(line_color[0],line_color[1],line_color[2])
    ctx.set_line_width(line_width)
    ctx.stroke()
#---------------------------------------------------------------------------------------------------#
def draw_text(ctx,text,coord,size,color=(0.0,0.0,0.0)):
    """ Draws text given its coordinates and size.

    Parameters
        ----------
        ctx : Cairo Context
            Context where the circle is going to be drawn.
        text : str
            Text to be written in Arial font.
        coord : tuple
            Tuple with x and y coordinate of the top-left corner of the line holding
            the text.
        size : float
            Height of the line holding the text.
        color : tuple, optional
            3- tuple containing the tex color in RGB format. Default is black (0.0,0.0,0.0).
    """
    ctx.set_source_rgb(color[0],color[1],color[2])
    ctx.set_font_size(size)
    ctx.select_font_face("Arial",
                         cr.FONT_SLANT_NORMAL,
                         cr.FONT_WEIGHT_NORMAL)
    ctx.move_to(coord[0],coord[1])
    ctx.show_text(text)
