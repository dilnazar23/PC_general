from win32com.client.gencache import EnsureDispatch, EnsureModule
from win32com.client import CastTo, constants
from win32com.client import gencache
import os, time, ctypes, array
import matplotlib.pyplot as plt
import numpy as np

# Notes
#
# The python project and script was tested with the following tools:
#       Python 3.4.3 for Windows (32-bit) (https://www.python.org/downloads/) - Python interpreter
#       Python for Windows Extensions (32-bit, Python 3.4) (http://sourceforge.net/projects/pywin32/) - for COM support
#       Microsoft Visual Studio Express 2013 for Windows Desktop (https://www.visualstudio.com/en-us/products/visual-studio-express-vs.aspx) - easy-to-use IDE
#       Python Tools for Visual Studio (https://pytools.codeplex.com/) - integration into Visual Studio
#
# Note that Visual Studio and Python Tools make development easier, however this python script should should run without either installed.

class PythonStandaloneApplication(object):
    class LicenseException(Exception):
        pass

    class ConnectionException(Exception):
        pass

    class InitializationException(Exception):
        pass

    class SystemNotPresentException(Exception):
        pass

    def __init__(self):
        # make sure the Python wrappers are available for the COM client and
        # interfaces
        gencache.EnsureModule('{EA433010-2BAC-43C4-857C-7AEAC4A8CCE0}', 0, 1, 0)
        gencache.EnsureModule('{F66684D7-AAFE-4A62-9156-FF7A7853F764}', 0, 1, 0)
        # Note - the above can also be accomplished using 'makepy.py' in the
        # following directory:
        #      {PythonEnv}\Lib\site-packages\wind32com\client\
        # Also note that the generate wrappers do not get refreshed when the
        # COM library changes.
        # To refresh the wrappers, you can manually delete everything in the
        # cache directory:
        #      {PythonEnv}\Lib\site-packages\win32com\gen_py\*.*
        
        self.TheConnection = EnsureDispatch("ZOSAPI.ZOSAPI_Connection")
        if self.TheConnection is None:
            raise PythonStandaloneApplication.ConnectionException("Unable to intialize COM connection to ZOSAPI")

        self.TheApplication = self.TheConnection.CreateNewApplication()
        if self.TheApplication is None:
            raise PythonStandaloneApplication.InitializationException("Unable to acquire ZOSAPI application")

        if self.TheApplication.IsValidLicenseForAPI == False:
            raise PythonStandaloneApplication.LicenseException("License is not valid for ZOSAPI use")

        self.TheSystem = self.TheApplication.PrimarySystem
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException("Unable to acquire Primary system")

    def __del__(self):
        if self.TheApplication is not None:
            self.TheApplication.CloseApplication()
            self.TheApplication = None

        self.TheConnection = None

    def OpenFile(self, filepath, saveIfNeeded):
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException("Unable to acquire Primary system")
        self.TheSystem.LoadFile(filepath, saveIfNeeded)

    def CloseFile(self, save):
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException("Unable to acquire Primary system")
        self.TheSystem.Close(save)

    def SamplesDir(self):
        if self.TheApplication is None:
            raise PythonStandaloneApplication.InitializationException("Unable to acquire ZOSAPI application")

        return self.TheApplication.SamplesDir

    def ExampleConstants(self):
        if self.TheApplication.LicenseStatus is constants.LicenseStatusType_PremiumEdition:
            return "Premium"
        elif self.TheApplication.LicenseStatus is constants.LicenseStatusType_ProfessionalEdition:
            return "Professional"
        elif self.TheApplication.LicenseStatus is constants.LicenseStatusType_StandardEdition:
            return "Standard"
        else:
            return "Invalid"


if __name__ == '__main__':
    zosapi = PythonStandaloneApplication()
    value = zosapi.ExampleConstants()
    
    if not os.path.exists(zosapi.TheApplication.SamplesDir + "\\API\\Python"):
        os.makedirs(zosapi.TheApplication.SamplesDir + "\\API\\Python")

    TheApplication = zosapi.TheApplication
    TheSystem = zosapi.TheSystem
    TheSystem.LoadFile("1550+508EFL_non_seqential-NONSEQ_V05_machine_learning.zmx",False)## put the entire path in here
    TheNCE = TheSystem.NCE
    suncave = TheNCE.GetObjectAt(19)
    detector = TheNCE.GetObjectAt(29)
    
    #! [e24s09_py]
    # Setup and run the ray trace
    NSCRayTrace = TheSystem.Tools.OpenNSCRayTrace()
    NSCRayTrace.SplitNSCRays = True
    NSCRayTrace.ScatterNSCRays = True
    NSCRayTrace.UsePolarization = True
    NSCRayTrace.IgnoreErrors = True
    NSCRayTrace.SaveRays = False
    NSCRayTrace.ClearDetectors(0) # clear any previous detector data
    
    baseTool = CastTo(NSCRayTrace, 'ISystemTool')
    baseTool.RunAndWaitForCompletion()
    baseTool.Close()
    #! [e24s09_py]

    tic = time.time()


    #! [e24s13_py]
    # changes default values for Detector Viewer
    # pltos the Incoherent Irradiance in False Color
    d5 = TheSystem.Analyses.New_Analysis(constants.AnalysisIDM_DetectorViewer)
    d5_set = d5.GetSettings()
    setting = CastTo(d5_set, 'IAS_DetectorViewer')
    setting.Detector.SetDetectorNumber(29)
    setting.ShowAs = constants.DetectorViewerShowAsTypes_FalseColor
    d5.ApplyAndWaitForCompletion()
    d5_results = d5.GetResults()
    results = CastTo(d5_results, 'IAR_')
    d5_values = results.GetDataGrid(0).Values
    # Save results to a CSV file
    output_csv = r"result.csv" ## put the entire path here?
    np.savetxt(output_csv, np.flipud(d5_values), delimiter=",", fmt="%.6f")
    print(f"Irradiance results saved to {output_csv}")

    #! [e24s13_py]
    plt.figure()
    plt.imshow(np.flipud(d5_values), cmap='plasma')
    plt.colorbar()
    
    toc = round(time.time() - tic, 3)
    print('Elapsed time is ' + str(toc) + ' seconds.')
    
    #! [e24s14_py]
    # saves current system in memory
    #TheSystem.Save()
    #! [e24s14_py]


    # This will clean up the connection to OpticStudio.
    # Note that it closes down the server instance of OpticStudio, so you for maximum performance do not do
    # this until you need to.
    del zosapi
    zosapi = None

    # place plt.show() after clean up to release OpticStudio from memory
    plt.show()
     


