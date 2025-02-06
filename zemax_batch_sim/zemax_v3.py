from win32com.client.gencache import EnsureDispatch, EnsureModule
from win32com.client import CastTo, constants
from win32com.client import gencache
import os, time, ctypes, array
import matplotlib.pyplot as plt
import numpy as np

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
    TheSystem.LoadFile("1550+508EFL_non_seqential-NONSEQ_V05_machine_learning.zmx",False) ## change the file position
    TheNCE = TheSystem.NCE
    suncave = TheNCE.GetObjectAt(19)
    detector = TheNCE.GetObjectAt(29)

    output_folder = "results"
    os.makedirs(output_folder, exist_ok=True)

    iter_num = 0
    #change the position every time and rerun the thing 4 times
    for i in range(0,3):
        for j in range(0,3):
            iter_num += 1
            suncave.XPosition = 144+i
            suncave.YPosition = 126+j
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
            d5.ApplyAndWaitForCompletion()
            results = CastTo(d5_results, 'IAR_')
            d5_values = results.GetDataGrid(1).Values.double
            #d5_results =results.Get
            # Save results to CSV
            csv_filename = os.path.join(output_folder, f"results_round_{iter_num}.csv")
            np.savetxt(csv_filename, np.flipud(d5_values), delimiter=",", fmt="%.6f")
            print(f"Saved detector results for iteration {iter_num} to {csv_filename}")
            
            #! [e24s13_py]
            plt.figure()
            plt.imshow(np.flipud(d5_values), cmap='plasma')
            plt.colorbar()
            plt.title(f"round {iter_num}")
            plot_filename = os.path.join(output_folder, f"detector_plot_iter_{iter_num}.png")
            plt.savefig(plot_filename)
            plt.close()
            print(f"Saved detector plot for iteration {i + 1} to {plot_filename}")
            toc = round(time.time() - tic, 3)
            print('Elapsed time is ' + str(toc) + ' seconds.')
    
    #! [e24s14_py]
    # saves current system in memory
    #TheSystem.Save()
    #! [e24s14_py]


    # This will clean up the connection to OpticStudio.
    # Note that it closes down the server instance of OpticStudio, so you for maximum performance do not do
    # this until you need to.
    # del zosapi
    # zosapi = None

    # place plt.show() after clean up to release OpticStudio from memory
    plt.show()
     


