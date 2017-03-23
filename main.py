# -*- coding: utf-8 -*-

import wx
import cStringIO
import os
import cv2
import math
import numpy
import pylab
from Queue import Queue
from threading import Thread
from perceptron import Perceptron
from helpers import Helpers
from plateextraction import PlateExtraction

import time


class MainWindow(wx.Frame):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.init_window()
        self.make_menu()
        self.make_layout()

        self.neural_network_learned = False
        self.image_opened = False
        self.imagesOpened = False

        self.selected = 0
        self.filenames = []
        self.images = []
        self.plates = []
        self.segmentedplates = []
        self.platenumbers = []

        self.perceptron = None

        self.queue = Queue()

    def init_window(self):

        self.SetSize((1024, 768))
        self.SetTitle(u"Kamil Piec - Wyodrębnianie i identyfikacja tablic rejestracyjnych")
        self.SetWindowStyle(wx.MINIMIZE_BOX | wx.SYSTEM_MENU | wx.CAPTION | wx.CLOSE_BOX)
        self.Centre()
        self.Show(True)

    def make_menu(self):

        menubar = wx.MenuBar()

        file_menu = wx.Menu()

        self.open_item = wx.MenuItem(file_menu, wx.ID_ANY,
                                     u'Otwórz obraz',
                                     u'Otwórz obraz do przetwarzania')
        self.learn_item = wx.MenuItem(file_menu, wx.ID_ANY,
                                      u'Naucz sieć neuronową',
                                      u'Naucz sieć neuronową')
        self.save_item = wx.MenuItem(file_menu, wx.ID_ANY,
                                     u'Zapisz stan sieci neuronowej',
                                     u'Zapisz stan sieci neuronowej do pliku')
        self.load_item = wx.MenuItem(file_menu, wx.ID_ANY,
                                     u'Wczytaj stan sieci neuronowej ',
                                     u'Wczytaj stan sieci neuronowej z pliku')
        self.end_item = wx.MenuItem(file_menu, wx.ID_EXIT,
                                    u'Zakończ', u'Zakończ aplikację')

        file_menu.AppendItem(self.open_item)
        file_menu.AppendSeparator()
        file_menu.AppendItem(self.learn_item)
        file_menu.AppendItem(self.save_item)
        file_menu.AppendItem(self.load_item)
        file_menu.AppendSeparator()
        file_menu.AppendItem(self.end_item)

        self.save_item.Enable(False)

        menubar.Append(file_menu, '&Plik')

        self.CreateStatusBar()

        self.SetMenuBar(menubar)
        self.Bind(wx.EVT_MENU, self.learn_neural_network, self.learn_item)
        self.Bind(wx.EVT_MENU, self.open_file, self.open_item)
        self.Bind(wx.EVT_MENU, self.save, self.save_item)
        self.Bind(wx.EVT_MENU, self.load, self.load_item)
        self.Bind(wx.EVT_MENU, self.on_quit, self.end_item)

    def make_layout(self):

        self.panel = wx.Panel(self)
        self.panel.SetBackgroundColour('#4e4e4e')

        self.mid_pan = wx.Panel(self.panel)

        self.mid_left_pan = wx.Panel(self.mid_pan)
        self.mid_left_pan.SetBackgroundColour('#ededed')

        self.mid_right_pan = wx.Panel(self.mid_pan)
        self.mid_right_pan.SetBackgroundColour('#ededed')

        self.listbox = wx.ListBox(self.mid_right_pan, style=wx.LB_NEEDED_SB | wx.LB_HSCROLL)

        self.Bind(wx.EVT_LISTBOX, lambda event: self.show_image(event, self.listbox.GetSelection()))

        h_box = wx.BoxSizer(wx.HORIZONTAL)
        h_box.Add(self.listbox, 1, wx.EXPAND | wx.ALL, 2)

        self.mid_right_pan.SetSizer(h_box)

        h_box = wx.BoxSizer(wx.HORIZONTAL)
        h_box.Add(self.mid_left_pan, 4, wx.EXPAND | wx.ALL, 2)
        h_box.Add(self.mid_right_pan, 1, wx.EXPAND | wx.ALL, 2)

        self.mid_pan.SetSizer(h_box)

        self.bot_pan = wx.Panel(self.panel)

        self.bot_left_pan = wx.Panel(self.bot_pan)
        self.bot_left_pan.SetBackgroundColour('#ededed')

        self.bot_mid_pan = wx.Panel(self.bot_pan)
        self.bot_mid_pan.SetBackgroundColour('#ededed')

        self.bot_right_pan = wx.Panel(self.bot_pan)
        self.bot_right_pan.SetBackgroundColour('#ededed')

        self.right_text = wx.StaticText(self.bot_right_pan, -1, "", (200, 10), style=wx.ALIGN_CENTRE)
        self.right_text.SetFont(wx.Font(25, wx.SWISS, wx.NORMAL, wx.BOLD))

        right_text_sizer = wx.BoxSizer(wx.VERTICAL)

        right_text_sizer.AddStretchSpacer()
        right_text_sizer.Add(self.right_text, 0, wx.CENTER)
        right_text_sizer.AddStretchSpacer()

        self.bot_right_pan.SetSizer(right_text_sizer)

        h_box = wx.BoxSizer(wx.HORIZONTAL)
        h_box.Add(self.bot_left_pan, 3, wx.EXPAND | wx.ALL, 1)
        h_box.Add(self.bot_mid_pan, 3, wx.EXPAND | wx.ALL, 1)
        h_box.Add(self.bot_right_pan, 3, wx.EXPAND | wx.ALL, 1)

        self.bot_pan.SetSizer(h_box)

        self.gauge = wx.Gauge(self.panel, range=1000, size=(250, 25))

        v_box = wx.BoxSizer(wx.VERTICAL)
        v_box.Add(self.mid_pan, 30, wx.EXPAND | wx.ALL, 2)
        v_box.Add(self.bot_pan, 5, wx.EXPAND | wx.ALL, 2)
        v_box.Add(self.gauge, 1, wx.EXPAND | wx.ALL, 2)

        self.panel.SetSizer(v_box)

    def open_file(self, e, filename=''):

        open_file_dialog = wx.FileDialog(self, u"Otwórz obraz", "", "",
                                         "Pliki graficzne (*.png;*.jpg)|*.png;*.jpg",
                                         wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE)

        if open_file_dialog.ShowModal() == wx.ID_CANCEL:
            return

        for i in open_file_dialog.GetPaths():
            self.listbox.Append(os.path.basename(i))
            self.filenames.append(i)
            self.images.append(pylab.imread(i, cv2.CV_8U))
            self.plates.append(None)
            self.segmentedplates.append(None)
            self.platenumbers.append(None)

        self.listbox.SetSelection(wx.NOT_FOUND)

    def show_image(self, e, index):

        self.selected = index

        try:
            data = open(self.filenames[index], "rb").read()
            stream = cStringIO.StringIO(data)
            bmp = wx.BitmapFromImage(wx.ImageFromStream(stream).Rescale(
                self.mid_left_pan.GetSize()[0] - 0,
                self.mid_left_pan.GetSize()[1] - 0))
            s_bmp = wx.StaticBitmap(self, -1, bmp, (4, 4))

            sizer = wx.BoxSizer()
            sizer.Add(s_bmp, 1, wx.EXPAND | wx.ALL, 1)

            self.mid_left_pan.SetSizer(sizer)

            if self.neural_network_learned:
                if self.plates[index] is None:
                    self.recognize()

                if self.platenumbers[index] is not None:
                    if len(self.platenumbers[index]) >= 3:
                        if self.platenumbers[index][2].isalpha():
                            self.platenumbers[index] = self.platenumbers[index][:3] + ' ' + self.platenumbers[index][3:]
                        else:
                            self.platenumbers[index] = self.platenumbers[index][:2] + ' ' + self.platenumbers[index][2:]

                    self.right_text.SetLabel(self.platenumbers[index])
                    self.bot_right_pan.Layout()
                else:
                    self.right_text.SetLabel('')
                    self.bot_right_pan.Layout()
                    dial = wx.MessageDialog(
                        self, u'Nie znaleziono numeru tablicy rejestracyjnej', u'Błąd', wx.OK | wx.ICON_ERROR)
                    dial.ShowModal()

                if self.plates[index] is not None:
                    width, height = self.bot_left_pan.GetSize()
                    res = cv2.resize(self.plates[index], (width, height), interpolation=cv2.INTER_AREA)
                    img = cv2.cvtColor(numpy.array(res, 'uint8'), cv2.COLOR_GRAY2RGB)
                    h, w = img.shape[:2]
                    wxbmp = wx.BitmapFromBuffer(w, h, img)
                    s_bmp = wx.StaticBitmap(self.bot_left_pan, -1, wxbmp, (10, 5))

                    box_left_sizer = wx.BoxSizer()
                    box_left_sizer.Add(s_bmp, 1, wx.EXPAND | wx.ALL, 1)

                    self.bot_left_pan.SetSizer(box_left_sizer)
                    self.bot_left_pan.Layout()

                if self.segmentedplates[index] is not None:
                    width, height = self.bot_mid_pan.GetSize()
                    res = cv2.resize(
                        numpy.array(self.segmentedplates[index], 'float64'),
                        (width, height),
                        interpolation=cv2.INTER_AREA)
                    img = cv2.cvtColor(numpy.array(res, 'uint8'), cv2.COLOR_GRAY2RGB)
                    h, w = img.shape[:2]
                    wxbmp = wx.BitmapFromBuffer(w, h, img)
                    s_bmp = wx.StaticBitmap(self.bot_mid_pan, -1, wxbmp, (10, 5))

                    box_mid_sizer = wx.BoxSizer()
                    box_mid_sizer.Add(s_bmp, 1, wx.EXPAND | wx.ALL, 1)

                    self.bot_mid_pan.SetSizer(box_mid_sizer)
                    self.bot_mid_pan.Layout()
                else:
                    for child in self.bot_mid_pan.GetChildren():
                        child.Destroy()
                    self.bot_mid_pan.Layout()
                    dial = wx.MessageDialog(
                        self, u'Nie udało się posegmentować tablicy rejestracyjnej', u'Błąd', wx.OK | wx.ICON_ERROR)
                    dial.ShowModal()
        except IOError:
            dial = wx.MessageDialog(
                self, u'Nie udało się otworzyć pliku', u'Błąd', wx.OK | wx.ICON_ERROR)
            dial.ShowModal()

    def learn_neural_network(self, e):

        self.learn_item.Enable(False)
        self.load_item.Enable(False)
        self.save_item.Enable(False)
        self.open_item.Enable(False)

        queue = Queue()

        self.thread1 = Thread(target=self.learn, args=(queue,))
        self.thread2 = Thread(target=self.check_progress, args=(queue,))

        self.thread1.start()
        self.thread2.start()

    def learn(self, queue):

        data = Helpers.learning2data()
        x = data[0]
        d = data[1]

        bad = True

        while bad:
            try:
                dial = wx.TextEntryDialog(
                    self, u'Podaj współczynnik uczenia \u03b7',
                    u'Podaj współczynnik uczenia \u03b7', defaultValue='0.9')
                dial.ShowModal()
                result = float(dial.GetValue())
                bad = False
            except ValueError:
                bad = True

        bad = True

        while bad:
            try:
                dial = wx.TextEntryDialog(
                    self, u'Podaj próg błędu Qmax', u'Podaj próg błędu Qmax', defaultValue='0.01')
                dial.ShowModal()
                result2 = float(dial.GetValue())
                bad = False
            except ValueError:
                bad = True

        self.perceptron = Perceptron(
            n=1700,
            m=35,
            x=x,
            d=d,
            eta=float(result),
            target=float(result2))

        self.weights = self.perceptron.learn(queue)

        try:
            dial = wx.MessageDialog(self, u'Nauczono sieć neuronową', u'Informacja', wx.OK | wx.ICON_INFORMATION)
            dial.ShowModal()
        except TypeError:
            return

        self.neural_network_learned = True
        self.save_item.Enable(True)
        self.learn_item.Enable(True)
        self.load_item.Enable(True)
        self.save_item.Enable(True)
        self.open_item.Enable(True)

    def check_progress(self, queue):

        self.gauge.SetValue(0)

        result = None
        while result is None:
            try:
                while queue.get() < 100:
                    self.gauge.SetValue(math.floor(queue.get() * 10))
                result = True
            except NameError:
                pass

        self.gauge.SetValue(0)

    def load(self, e):

        open_file_dialog = wx.FileDialog(self, u"Otwórz plik", "", "",
                                         "Pliki tekstowe (*.txt)|*.txt", wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

        if open_file_dialog.ShowModal() == wx.ID_CANCEL:
            return

        try:
            loaded = numpy.loadtxt(open_file_dialog.GetPath())

            self.neural_network_learned = True
            self.save_item.Enable(True)

            self.weights = loaded

            self.perceptron = Perceptron(
                n=1700,
                m=35,
                x=None,
                d=None,
                eta=None,
                target=None)

            dial = wx.MessageDialog(self, u'Wczytano stan sieci', u'Informacja', wx.OK | wx.ICON_INFORMATION)
            dial.ShowModal()
        except ValueError:
            dial = wx.MessageDialog(
                self, u'Błędny plik\nNie udało się wczytać stanu sieci', u'Błąd', wx.OK | wx.ICON_ERROR)
            dial.ShowModal()

    def save(self, e):

        save_file_dialog = wx.FileDialog(self, u"Zapisz plik", "", "",
                                         "Pliki tekstowe (*.txt)|*.txt", wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)

        if save_file_dialog.ShowModal() == wx.ID_CANCEL:
            return

        numpy.savetxt(save_file_dialog.GetPath(), self.weights)

        dial = wx.MessageDialog(self, u'Zapisano stan sieci', u'Informacja', wx.OK | wx.ICON_INFORMATION)
        dial.ShowModal()

    def on_quit(self, e):

        self.Close()

        raise IndexError

    def recognize(self):

        found = PlateExtraction.find_plate(cv2.cvtColor(self.images[self.selected], cv2.COLOR_RGB2GRAY))

        if found is not None:
            self.plates[self.selected] = found

            segments = PlateExtraction.segment_plate(found)

            if len(segments) > 0:
                training = []

                fullsegments = None

                for segment in segments:
                    segment = Helpers.crop_image(segment)
                    resized = cv2.resize(segment, (34, 50), interpolation=cv2.INTER_AREA)

                    ret, threshold = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    border = cv2.copyMakeBorder(threshold, 6, 6, 3, 3, cv2.BORDER_CONSTANT, value=[127, 127, 127])

                    fullsegments = numpy.concatenate((fullsegments, border), axis=1) \
                        if fullsegments is not None else border

                    training.append(threshold.ravel())

                plate_number = self.perceptron.test(self.weights, numpy.asarray(training))

                self.segmentedplates[self.selected] = fullsegments
                self.platenumbers[self.selected] = plate_number
        else:
            dial = wx.MessageDialog(
                self, u'Nie udało się znaleźć tablicy rejestracyjnej', u'Błąd', wx.OK | wx.ICON_ERROR)
            dial.ShowModal()


if __name__ == '__main__':
    app = wx.App()
    MainWindow(None)
    app.MainLoop()
